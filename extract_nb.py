import json

with open('Lung_Cancer_Prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print all cells with outputs
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        outputs = cell.get('outputs', [])
        if src.strip() and outputs:
            print(f"=== CELL {i} ===")
            print("SOURCE:")
            print(src)
            print("OUTPUTS:")
            for out in outputs:
                if out.get('output_type') == 'stream':
                    print(''.join(out.get('text', [])))
                elif out.get('output_type') in ('execute_result', 'display_data'):
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        print(''.join(data['text/plain']))
            print()
