import os
import sys

path = os.path.dirname(os.path.abspath(__file__))

# Iterate through each custom agent folder, e.g. Agent1, Agent2, ...
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        # Iterate through all python files, except for package marker __init__.py
        for py in [f[:-3] for f in os.listdir(os.path.join(path, folder)) if f.endswith('.py') and f != '__init__.py']:
            mod = __import__('.'.join([__name__, folder, py]), fromlist=[py])
            classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
            for cls in classes:
                # Add all found classes as global attributes
                setattr(sys.modules[__name__], cls.__name__, cls)