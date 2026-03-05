import torch
from collections import defaultdict

def extract_regions(model, X, mass_threshold=1):
    """
    Extracts regions defined by gate patterns and returns their mass.
    """
    patterns = model.gate_pattern(X)
    
    region_map = defaultdict(list)
    for i in range(X.shape[0]):
        pattern_tuple = tuple(patterns[i].tolist())
        region_map[pattern_tuple].append(i)
        
    extracted_regions = []
    for pattern_id, point_indices in region_map.items():
        mass = len(point_indices)
        if mass >= mass_threshold:
            extracted_regions.append({
                "region_id": pattern_id,
                "indices": point_indices,
                "mass": mass
            })
            
    return extracted_regions
