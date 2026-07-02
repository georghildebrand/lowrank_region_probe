import torch
from collections import defaultdict

def extract_regions(model, X, cell_ids=None, mass_threshold=1):
    """
    Extracts regions defined by gate patterns and returns their mass.
    If cell_ids is provided, computes purity = dominant cell_id fraction.
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
            region_dict = {
                "region_id": pattern_id,
                "indices": point_indices,
                "mass": mass
            }
            
            # Compute purity if cell_ids are provided
            if cell_ids is not None:
                region_cells = cell_ids[point_indices]
                # Drop unassigned points (-1) — bincount rejects negatives
                region_cells = region_cells[region_cells >= 0]
                if len(region_cells) > 0:
                    counts = torch.bincount(region_cells)
                    region_dict["purity"] = counts.max().item() / mass
                else:
                    region_dict["purity"] = 0.0
                
            extracted_regions.append(region_dict)
            
    return extracted_regions
