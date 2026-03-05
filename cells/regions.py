import torch
import numpy as np
from collections import defaultdict

def gate_patterns(model, X):
    """Returns Boolean tensor [N, hidden] of gate activations."""
    return model.gate_pattern(X)

def hash_patterns(patterns):
    """
    Fast hashing of Boolean patterns.
    Uses bitpacking into int64 blocks for performance.
    """
    N, M = patterns.shape
    # Pack bits into uint8 first (native torch support)
    # Then view as larger integers or just use as is for grouping
    # For robust unique identification across any M, we use a string-like or byte-based hash 
    # but for N=10k, m=64, bitpacking into a single int64 is perfect.
    
    if M <= 64:
        # Manual bitpacking into uint64
        packed = torch.zeros(N, dtype=torch.int64, device=patterns.device)
        for i in range(M):
            packed |= (patterns[:, i].long() << i)
        return packed.cpu().numpy()
    else:
        # For M > 64, we fallback to more complex grouping or multiple int64s
        # But per manual, we target m=64
        packed_blocks = []
        for i in range(0, M, 64):
            block_m = min(64, M - i)
            block = torch.zeros(N, dtype=torch.int64, device=patterns.device)
            for j in range(block_m):
                block |= (patterns[:, i + j].long() << j)
            packed_blocks.append(block.cpu().numpy())
        
        # Combine blocks into a single unique ID per row
        # A simple way is to use the tuples of blocks as keys
        hashes = np.empty(N, dtype=object)
        for idx in range(N):
            hashes[idx] = tuple(pb[idx] for pb in packed_blocks)
        return hashes

def extract_cells(X, pattern_hash, min_mass=10):
    """
    Groups points by their pattern hash.
    Returns cell mapping and summary stats.
    """
    N = len(pattern_hash)
    groups = defaultdict(list)
    for i in range(N):
        groups[pattern_hash[i]].append(i)
        
    cell_ids = np.full(N, -1, dtype=int)
    cells = []
    
    cell_idx = 0
    # Sort by mass descending for stable indexing
    sorted_hashes = sorted(groups.keys(), key=lambda h: len(groups[h]), reverse=True)
    
    for h in sorted_hashes:
        indices = groups[h]
        mass = len(indices)
        if mass >= min_mass:
            cell_ids[indices] = cell_idx
            cells.append({
                "cell_index": cell_idx,
                "mass": mass,
                "member_indices": indices,
                "pattern_hash": h
            })
            cell_idx += 1
            
    return cell_ids, cells
