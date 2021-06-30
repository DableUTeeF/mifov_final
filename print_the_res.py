import json
a = json.load(open('./evaluate.json'))


for name in a:
    print('    \cline{1-7}', name, end=' ')
    b = a[name]['weighted_map']
    print(f"& {b['0.3']:.3f} & {b['0.5']:.3f} & {b['0.7']:.3f} & {b['0.9']:.3f}", end=' ')
    print('&', f"{1 / a[name]['time']:.3f}", '\\\\')
