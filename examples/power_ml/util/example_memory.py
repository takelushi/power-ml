"""Example of memory."""

from power_ml.util.memory import get_process_memory


def show_size():
    """Show process memory size."""
    print('Process:')
    print('    {:.0f} B'.format(get_process_memory()))
    print('    {:.0f} KB'.format(get_process_memory(unit='KB')))
    print('    {:.2f} MB'.format(get_process_memory(unit='MB')))
    print('    {:.2f} GB'.format(get_process_memory(unit='GB')))


show_size()

# Create objects
a = 123.456
b = 100
c = 'a'
d = 'a' * 100
e = 'a' * 10000
f = 'a' * 1000000
l1 = [1, 2, 3]
l2 = ['aaaa', 'b', 'c']
d1 = {str(v): v for v in range(1000)}
d12 = {str(v): 'a' * 100000 for v in range(1000)}

show_size()
