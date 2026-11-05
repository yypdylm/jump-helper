"""Bound normal-run image retention, without importing image or input libraries.

Only completed steps older than the recent ring can be compacted. The current
and previous scene files used by recognition/landing checks are never touched.
"""
import gzip
import json
from pathlib import Path
import re

from data_manager import atomic_json, sha


class RunLogStore:
    def __init__(self, directory, profile='compact', recent=5, samples=12):
        if profile not in ('compact', 'full') or recent < 2 or samples < 1:
            raise ValueError('Invalid log retention policy')
        self.directory=Path(directory); self.profile=profile; self.recent=recent; self.samples=samples
        self.protected=set(); self.sample_steps=[]; self.seen=set(); self.compacted=set()
        self.completed=0

    @staticmethod
    def novelty(scene):
        p=scene.get('platform', {}); width=scene.get('image_size',[400])[0]
        # A bounded geometric category, not a claim of visual/semantic equivalence.
        outline=p.get('outline', [])
        span=max((v[0] for v in outline),default=0)-min((v[0] for v in outline),default=0)
        size=min(10,max(0,int(10*span/max(1,width))))
        return (p.get('detector','unknown'),p.get('shape','unknown'),size)

    def after_step(self, step, payload):
        if self.profile=='full':return
        self.completed=step
        key=self.novelty(payload.get('scene',{}))
        if key not in self.seen:
            self.seen.add(key);self.protected.add(step)
        # All recovered/exceptional decisions remain detailed, regardless of sample quota.
        if payload.get('recovery') or payload.get('error'):
            self.protected.add(step)
        if step==1 or step%25==0:
            self.sample_steps.append(step)
            self.sample_steps=self.sample_steps[-self.samples:]
        keep=self.protected | set(self.sample_steps)
        cutoff=step-self.recent
        for candidate in range(1,cutoff+1):
            if candidate not in keep and candidate not in self.compacted:
                self._compact(candidate)
                self.compacted.add(candidate)
        atomic_json(self.directory/'retention.json',{'profile':self.profile,'recent_steps':self.recent,
                    'completed':step,'sample_steps':self.sample_steps,'detailed_steps':sorted(self.protected),
                    'compacted_steps':sorted(self.compacted),'index':'retention-index.jsonl'})

    def _compact(self, step):
        prefix=f'{step:04d}-'
        for path in sorted(self.directory.glob(prefix+'*')):
            if path.is_symlink() or not path.is_file():continue
            # Strict filename boundary prevents step 1 matching step 10, and no other tree is visited.
            if not re.fullmatch(r'\d{4,}-[\w.-]+',path.name):continue
            if path.suffix not in ('.png','.json','.txt'):continue
            data=path.read_bytes()
            action='remove-normal-image' if path.suffix=='.png' else 'gzip-lossless'
            if path.suffix=='.txt' and data:
                # Nonempty worker output may be an unexpected warning; retain it.
                continue
            entry={'path':path.name,'sha256':sha(data),'bytes':len(data),'step':step,'action':action}
            if action=='gzip-lossless':
                destination=path.with_suffix(path.suffix+'.gz')
                packed=gzip.compress(data,mtime=0)
                if gzip.decompress(packed)!=data:raise ValueError('Compression mismatch')
                temp=destination.with_suffix(destination.suffix+'.tmp');temp.write_bytes(packed);temp.replace(destination)
                entry['replacement']=destination.name
            # Record provenance before unlink. Unexpected errors leave data intact or recoverable.
            with (self.directory/'retention-index.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(entry,ensure_ascii=False)+'\n');stream.flush()
            path.unlink()
