import sys

if __package__ is not None:
    sys.path.insert(1,sys.path[0]+'/'+__package__)

# print(sys.path)

# Perform all local module imports