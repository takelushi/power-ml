"""Example of version."""

from power_ml.util.version import get_versions, show_versions

print('Default tagetse')
for ver in get_versions():
    print(ver)

targets = {
    'Poetry': 'poetry',
}

print('Custom targets')
for ver in get_versions(targets=targets):
    print(ver)

# Print versions.
show_versions()
