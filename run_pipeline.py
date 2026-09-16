import argparse
import traceback
from pathlib import Path
import yaml
from src.pipeline import run

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--profile',choices=['smoke','assignment'],default='smoke'); p.add_argument('--config',default='config.yaml'); a=p.parse_args()
    try:
        c=yaml.safe_load(Path(a.config).read_text(encoding='utf-8')); result=run(c,a.profile); print('Completed',a.profile,result, flush=True)
    except Exception:
        Path('outputs/audit').mkdir(parents=True, exist_ok=True)
        Path('outputs/audit/run_failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
