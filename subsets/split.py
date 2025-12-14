import json
import os

def keep_only_high_risk(output_path: str) -> None:
    all_high_risk_ids = []
    for file in os.listdir('./dataset'):
        if not file.endswith('.json'):
            continue
        with open(os.path.join('./dataset', file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data["Impact"] == "HIGH":
            all_high_risk_ids.append(file.split('.')[0])

    with open(output_path, 'w') as f:
        json.dump(all_high_risk_ids, f)
    
    print(f"Saved {len(all_high_risk_ids)} high risk IDs to {output_path}")

def random_500_high(output_path: str) -> None:
    with open('./subsets/high_risk_ids.json', 'r') as f:
        all_high_risk_ids = json.load(f)
    
    selected_ids = all_high_risk_ids[:500]

    with open(output_path, 'w') as f:
        json.dump(selected_ids, f)
    
    print(f"Saved {len(selected_ids)} high risk IDs to {output_path}")

if __name__ == "__main__":
    keep_only_high_risk('./subsets/high_risk_ids.json')
    random_500_high('./subsets/random_500_high_risk_ids.json')
