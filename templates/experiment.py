import json, platform, importlib.metadata as md
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
run=Path('outputs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
run.mkdir(parents=True,exist_ok=False)
versions={}
for pkg in ['numpy','pandas','scikit-learn','tensorflow','keras']:
    try: versions[pkg]=md.version(pkg)
    except md.PackageNotFoundError: versions[pkg]=None
config={'seed':42,'metric':'MAE','model':'constant demo','python':platform.python_version(),'versions':versions}
y=np.array([1.,2.,3.]); pred=np.ones_like(y)*2
config['score']=float(np.mean(abs(y-pred)))
(run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
np.save(run/'pred.npy',pred)
print(run,config['score'])
