import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P_loaded = False
cache_c_new = False

def apply_AlphaEdit_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: AlphaEditHyperParams,
        copy=False,
        return_orig_weights=False,
        cache_template: Optional[str] = None,
        keep_original_weight=False,
        **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
    global P, P_loaded, cache_c, cache_c_new

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    device = torch.device(torch_device_alias(hparams.device))

    if not os.path.exists(hparams.P_loc):
        print(f"The null-space projection matrix P does not exist. Calculating now...")

        # Get the weight matrix for the output layer
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        P_files = []

        for i, layer in enumerate(hparams.layers):
            print(f"Calculating projection matrix for layer {i + 1}/{len(hparams.layers)}...")

            # Determine the shape of the projection matrix based on the model type
            if "llama" in hparams.model_name.lower() or "gpt-j-6b" in hparams.model_name.lower():
                layer_shape = (W_out.shape[1], W_out.shape[1])
            elif "gpt2-xl" in hparams.model_name.lower():
                layer_shape = (W_out.shape[0], W_out.shape[0])
            else:
                raise ValueError(f"Unknown model type: {hparams.model_name}")

            # Initialize a zero matrix with the correct shape on the desired device
            layer_P = torch.zeros(layer_shape, device=torch_device_alias(hparams.device))

            # Calculate the actual projection matrix and assign it to the initialized matrix
            calculated_P = get_project(model, tok, layer, hparams).to(torch_device_alias(hparams.device))

            # Force the correct shape by copying the calculated matrix
            layer_P[:calculated_P.shape[0], :calculated_P.shape[1]] = calculated_P

            # Save the current layer's projection matrix to a file (CPU)
            layer_file = f"null_space_project_layer_{i}.pt"
            torch.save(layer_P.to("cpu"), layer_file)
            P_files.append(layer_file)

            # Clear memory for the next layer
            del layer_P, calculated_P

            # Clear MPS memory if needed
            torch[torch_device_alias(hparams.device)].empty_cache()

        # Combine all layer matrices into one tensor and save
        print("Combining layer matrices into a single file...")
        P_combined = torch.stack([torch.load(f, map_location="cpu") for f in P_files], dim=0)
        torch.save(P_combined, hparams.P_loc)

        # Clean up intermediate files
        for f in P_files:
            os.remove(f)

        P_loaded = True
    elif not P_loaded:
        # Load the existing projection matrix from disk
        P = torch.load(hparams.P_loc, map_location=device)
        P_loaded = True

    if not cache_c_new:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if "llama" in hparams.model_name.lower() or "gpt-j-6b" in hparams.model_name.lower():
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        elif "gpt2-xl" in hparams.model_name.lower():
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        del W_out
        cache_c_new = True

    deltas = execute_AlphaEdit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            upd_matrix = upd_m.to(torch_device_alias(hparams.device))
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()
            print(f"Updated weights for {w_name}")

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
        requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
        print(
            f"Executing AlphaEdit algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(torch_device_alias(hparams.device)))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        # kressaty removed to support mps, TODO: MAKE FALLBACK
        upd_matrix = torch.linalg.solve(
            P[i,:,:].to("cpu") @ (layer_ks.to("cpu") @ layer_ks.T.to("cpu") + cache_c[i,:,:].to("cpu")) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cpu"),
            P[i,:,:].to("cpu") @ layer_ks.to("cpu") @ resid.T.to("cpu")
        )

        upd_matrix = upd_matrix.to(torch_device_alias(hparams.device))

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                upd_matrix.detach().cpu()
            )
        
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch[torch_device_alias(hparams.device)].empty_cache()
    
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    # todo: do we need to remove this?
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    
    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        layer_name: str,
        mom2_dataset: str,
        mom2_n_samples: int,
        mom2_dtype: str,
        inv: bool = False,
        force_recompute: bool = False,
        hparams=None,
) -> torch.Tensor:
    """
    Efficiently retrieves or computes the covariance matrix in chunks.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
            batch_tokens=2048,
        )

        # Compute the covariance matrix in chunks
        full_cov = stat.mom2.moment().float().to("cpu")
        chunk_size = 512  # Adjust based on available memory
        rows, cols = full_cov.shape
        cov = torch.zeros((rows, cols), dtype=full_cov.dtype)

        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            cov[start:end] = full_cov[start:end] @ full_cov.T

        COV_CACHE[key] = cov

    return (
        torch.inverse(COV_CACHE[key].to(torch_device_alias(hparams.device))) if inv else COV_CACHE[key].to(torch_device_alias(hparams.device))
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by AlphaEdit does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_project(model, tok, layer, hparams):
    """
    Computes the null-space projection matrix for a given layer.
    Optimized for memory efficiency with chunked covariance computation.
    """

    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
        hparams=hparams,
    ).to(torch_device_alias(hparams.device))  # Keep on MPS for efficiency

    # Compute SVD on MPS
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)

    # Threshold small singular values
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(f"Small singular values count: {len(small_singular_indices)}")

    # Compute projection matrix
    U_reduced = U[:, small_singular_indices]
    projection_matrix = U_reduced @ U_reduced.T

    return projection_matrix.to("cpu")  # Return to CPU for saving