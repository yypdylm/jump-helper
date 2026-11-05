"""Local, lossless log archive and provenance catalog. No game input or network."""
import argparse
import ast
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def files_under(root):
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlink in data tree: {p}')
        if p.is_file():
            yield p


def fixture_references(root):
    """Literal references plus dynamically constructed names; uncertainty is retained."""
    refs = defaultdict(list)
    fixtures = list((root / 'testdata').glob('*.png'))
    for test in sorted(root.glob('test_*.py')):
        tree = ast.parse(test.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
                source = ast.get_source_segment(test.read_text(), node)
                for fixture in fixtures:
                    stem = fixture.stem
                    dynamic = stem.rsplit('-', 1)[0] if stem.endswith(('-before', '-after', '-previous')) else stem
                    if fixture.name in source or dynamic in source:
                        refs[fixture.name].append(f'{test.name}:{node.name}')
    return refs


def audit(root, output):
    import cv2
    import numpy as np
    root = Path(root).resolve()
    refs = fixture_references(root)
    records, by_hash, by_pixels, near = [], defaultdict(list), defaultdict(list), []
    thumbs = []
    for directory in ('logs', 'testdata'):
        for p in files_under(root / directory):
            data = p.read_bytes(); st = p.stat(); rel = p.relative_to(root).as_posix()
            item = {'path': rel, 'sha256': sha(data), 'bytes': len(data),
                    'allocated_bytes': st.st_blocks * 512, 'mtime_ns': st.st_mtime_ns,
                    'kind': 'fixture' if directory == 'testdata' else 'runtime',
                    'tests': refs.get(p.name, []) if directory == 'testdata' else []}
            by_hash[item['sha256']].append(rel)
            if p.suffix.lower() == '.png':
                im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if im is not None:
                    item['shape'] = list(im.shape)
                    item['pixel_sha256'] = sha(str(im.shape).encode() + im.tobytes())
                    by_pixels[item['pixel_sha256']].append(rel)
                    small = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (9, 8))
                    bits = np.packbits(small[:, 1:] > small[:, :-1]).tobytes()
                    item['dhash'] = bits.hex()
                    # Only raw scene/fixture frames, not annotations or desktop images.
                    if directory == 'testdata' or p.name.endswith('-scene.png'):
                        thumbs.append((rel, int.from_bytes(bits, 'big'), item['pixel_sha256']))
            records.append(item)
    for i, (a, ha, pa) in enumerate(thumbs):
        for b, hb, pb in thumbs[:i]:
            distance = (ha ^ hb).bit_count()
            if pa != pb and distance <= 2:
                near.append({'a': a, 'b': b, 'hamming_distance': distance,
                             'action': 'retain; similarity is not semantic equivalence'})
    exact = [v for v in by_hash.values() if len(v) > 1]
    pixel = [v for v in by_pixels.values() if len(v) > 1]
    sizes = {x['path']: x['bytes'] for x in records}
    report = {'schema': 1, 'root': str(root), 'files': records,
              'exact_duplicate_groups': exact, 'pixel_duplicate_groups': pixel,
              'near_candidates': near,
              'summary': {'files': len(records), 'bytes': sum(x['bytes'] for x in records),
                          'allocated_bytes': sum(x['allocated_bytes'] for x in records),
                          'png_files': len([x for x in records if 'pixel_sha256' in x]),
                          'duplicate_extra_files': sum(len(v)-1 for v in exact),
                          'duplicate_extra_bytes': sum(sum(sizes[n] for n in v[1:]) for v in exact),
                          'near_candidate_pairs': len(near),
                          'fixtures': len(refs),
                          'unresolved_fixtures': [p.name for p in (root/'testdata').glob('*.png') if p.name not in refs]}}
    atomic_json(output, report)
    print(json.dumps(report['summary'], ensure_ascii=False))
    return report


def create_archive(source, archive):
    """Capture all bytes with original paths, modes and timestamps; verify before return."""
    source, archive = Path(source).resolve(), Path(archive).resolve()
    if archive.is_relative_to(source):
        raise ValueError('Archive must be outside its source tree')
    if archive.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    entries, seen = [], set()
    temp = archive.with_suffix('.partial')
    try:
        with zipfile.ZipFile(temp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in files_under(source):
                data = p.read_bytes(); st = p.stat(); digest = sha(data)
                entries.append({'path': p.relative_to(source).as_posix(), 'sha256': digest,
                                'bytes': len(data), 'mtime_ns': st.st_mtime_ns, 'mode': st.st_mode & 0o777})
                if digest not in seen:
                    z.writestr('objects/'+digest, data); seen.add(digest)
            manifest = {'schema': 1, 'source': str(source), 'entries': entries,
                        'created_at': datetime.now().astimezone().isoformat()}
            z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False))
        verify_archive(temp)
        # Detect concurrent modification, including newly created or removed files.
        current = {p.relative_to(source).as_posix(): sha(p.read_bytes()) for p in files_under(source)}
        if current != {e['path']: e['sha256'] for e in entries}:
            raise RuntimeError('Source changed during archival; source left untouched')
        os.replace(temp, archive)
        return manifest
    finally:
        if temp.exists(): temp.unlink()


def verify_archive(path):
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('manifest.json'))
        checked = set()
        for e in manifest['entries']:
            rel = Path(e['path'])
            if rel.is_absolute() or '..' in rel.parts or not rel.parts:
                raise ValueError('Unsafe archived path')
            if e['sha256'] not in checked:
                data = z.read('objects/'+e['sha256'])
                if sha(data) != e['sha256'] or len(data) != e['bytes']:
                    raise ValueError('Archive checksum mismatch')
                checked.add(e['sha256'])
        return manifest


def restore_archive(archive, destination):
    """Restore into a new/empty directory only; no overwrite of a running project."""
    destination = Path(destination)
    manifest = verify_archive(archive)
    if destination.is_symlink():
        raise ValueError('Restore destination cannot be a symlink')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Restore destination must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for e in manifest['entries']:
            p = destination/e['path']; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read('objects/'+e['sha256']))
            os.chmod(p, e['mode']); os.utime(p, ns=(e['mtime_ns'], e['mtime_ns']))
    for e in manifest['entries']:
        if sha((destination/e['path']).read_bytes()) != e['sha256']:
            raise ValueError('Restored file checksum mismatch')
    return len(manifest['entries'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p=sub.add_parser('audit');p.add_argument('--root',type=Path,default=Path(__file__).parent);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('archive');p.add_argument('source',type=Path);p.add_argument('archive',type=Path)
    p=sub.add_parser('verify');p.add_argument('archive',type=Path)
    p=sub.add_parser('restore');p.add_argument('archive',type=Path);p.add_argument('destination',type=Path)
    a=parser.parse_args()
    if a.command=='audit':audit(a.root,a.output)
    elif a.command=='archive':print({'files':len(create_archive(a.source,a.archive)['entries']), 'source_untouched':True})
    elif a.command=='verify':print({'verified_files':len(verify_archive(a.archive)['entries'])})
    else:print({'restored_files':restore_archive(a.archive,a.destination)})


if __name__=='__main__':main()
