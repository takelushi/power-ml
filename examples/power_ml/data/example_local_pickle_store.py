"""Example of local pickle store."""

from power_ml.data.local_pickle_store import LocalPickleStore

store = LocalPickleStore('./tmp/store')

# Save.
obj = 123.456
name = store.save(obj)
print('Object name: {}'.format(name))

del obj
try:
    print(obj)  # type: ignore
except NameError:
    pass

# Load.
obj, meta = store.load_with_meta(name)
print('Object: {}'.format(obj))
print('Meta: {}'.format(meta))

# Save exist not ok.
# Raise FileExistsError.
try:
    name = store.save(obj, exist_ok=False)
except FileExistsError:
    pass

# Delete
store.delete(name)
