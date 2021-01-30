"""Example of S3 pickle store."""

import boto3

from power_ml.data.s3_pickle_store import S3PickleStore

bucket = list(boto3.resource('s3').buckets.all())[0].name
print('Bucket: {}'.format(bucket))
store = S3PickleStore(bucket, './tmp/s3', cache=False)

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

exit(0)
