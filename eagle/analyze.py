import json
import os
import random
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration Section ---
# Path to the JSON file containing all test prompts, their responses, and eagle_ids
INPUT_RESULTS_FILE = './data/vicuna.json'

# Tokenizer mapping: short name -> full path
TOKENIZER_MAP = {
    'vicuna': '/home/yanxing/data/models/vicuna-13b-v1.3',
    'llama3': '/home/yanxing/data/models/Meta-Llama-3.1-8B-Instruct',
    'deepseek': '/home/yanxing/data/models/DeepSeek-R1-Distill-Llama-8B',
    'qwen3': '/home/yanxing/data/models/Qwen3-0.6B' # Even if not src, can be a target
}

# List of model keys to analyze (these models must be defined in TOKENIZER_MAP
# and have corresponding INPUT_RESULTS_FILE)
ANALYSIS_MODELS = ['vicuna', 'llama3', 'deepseek']

# Language categories and their display labels
LANG_CATEGORIES_KEYS = ['zh', 'en', 'multi']
LANG_CATEGORIES_DISPLAY_LABELS = ['Chinese', 'English', 'Multilingual']
OVERALL_CATEGORY_KEY = 'overall'
OVERALL_CATEGORY_DISPLAY_LABEL = 'Overall Average'

# --- Utility Functions ---
def get_model_display_name(model_key: str) -> str:
    """Returns the short name for display."""
    return model_key

_tokenizers_cache = {} # Cache for loaded tokenizers to avoid redundant loading

def get_tokenizer(model_key: str):
    """
    Loads and caches a tokenizer using TOKENIZER_MAP.
    """
    if model_key not in TOKENIZER_MAP:
        print(f"Error: Model key '{model_key}' does not exist in TOKENIZER_MAP.")
        return None

    model_path = TOKENIZER_MAP[model_key]
    if model_key not in _tokenizers_cache:
        print(f"Loading tokenizer: {model_path} (Key: {model_key})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            _tokenizers_cache[model_key] = tokenizer
            print(f"Tokenizer '{model_key}' loaded successfully.")
        except Exception as e:
            print(f"Error: Could not load tokenizer '{model_path}' (Key: {model_key}): {e}")
            return None
    return _tokenizers_cache[model_key]

# --- Core Analysis Function ---
def analyze_single_src_model_results(
    results_file: str,
    src_model_key: str,
    all_model_keys: list[str] # All possible model keys (src and targets)
) -> dict:
    """
    Analyzes speculative decoding results for a single source model,
    calculating actual speedup and estimated speedup for other models.
    """
    tokenizer_src = get_tokenizer(src_model_key)
    if not tokenizer_src:
        return {}

    tokenizers_target = {} # Stores {target_model_key: tokenizer_object}
    # Target models are all models except the src_model_key
    target_model_keys = [key for key in all_model_keys if key != src_model_key]
    for key in target_model_keys:
        tokenizer = get_tokenizer(key)
        if tokenizer:
            tokenizers_target[key] = tokenizer
        else:
            print(f"Warning: Could not load target tokenizer '{key}', skipping this model.")

    if not tokenizers_target and len(all_model_keys) > 1:
        print("Error: No valid target tokenizers available for comparison.")

    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found.")
        return {}
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            all_results_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{results_file}': {e}")
        return {}
    except Exception as e:
        print(f"Error: An unknown error occurred while reading file '{results_file}': {e}")
        return {}

    all_src_actual_speedups = []
    all_target_estimated_speedups_by_model = {key: [] for key in tokenizers_target.keys()}

    categorized_analysis_results = { lang: [] for lang in LANG_CATEGORIES_KEYS }
    categorized_speedup_accumulators = {
        lang: {'src_actual': [], 'target_estimated': {key: [] for key in tokenizers_target.keys()}}
        for lang in LANG_CATEGORIES_KEYS
    }

    for lang_category_raw, prompts_list in all_results_data.items():
        lang_category = lang_category_raw if lang_category_raw != 'mix' else 'multi' # Handle 'mix' compatibility
        if lang_category not in LANG_CATEGORIES_KEYS:
            print(f"Warning: Unknown language category '{lang_category_raw}' found in data, skipping.")
            continue

        for prompt_data in prompts_list:
            prompt_id = prompt_data.get("id", "unknown_id")
            eagle_ids_log = prompt_data.get("eagle_ids")

            if not eagle_ids_log or len(eagle_ids_log) < 2:
                continue

            speculative_steps_tokens = eagle_ids_log[1:]

            if not speculative_steps_tokens:
                continue

            src_total_tokens_accepted = 0
            for step_tokens in speculative_steps_tokens:
                src_total_tokens_accepted += len(step_tokens)

            src_actual_speedup_this_case = src_total_tokens_accepted / len(speculative_steps_tokens)
            all_src_actual_speedups.append(src_actual_speedup_this_case)
            categorized_speedup_accumulators[lang_category]['src_actual'].append(src_actual_speedup_this_case)

            target_estimated_speedups_this_case = {}
            for target_model_key, tokenizer_target_obj in tokenizers_target.items():
                target_total_tokens_equivalent = 0
                for step_tokens_src in speculative_steps_tokens:
                    decoded_text_segment = tokenizer_src.decode(step_tokens_src, skip_special_tokens=True)
                    if not decoded_text_segment.strip():
                        target_tokens_equivalent_len = 0
                    else:
                        target_tokens_equivalent = tokenizer_target_obj.encode(decoded_text_segment, add_special_tokens=False)
                        target_tokens_equivalent_len = len(target_tokens_equivalent)

                    target_total_tokens_equivalent += target_tokens_equivalent_len

                if len(speculative_steps_tokens) > 0:
                    target_estimated_speedup = target_total_tokens_equivalent / len(speculative_steps_tokens)
                else:
                    target_estimated_speedup = 0

                target_estimated_speedups_this_case[target_model_key] = target_estimated_speedup
                all_target_estimated_speedups_by_model[target_model_key].append(target_estimated_speedup)
                categorized_speedup_accumulators[lang_category]['target_estimated'][target_model_key].append(target_estimated_speedup)

            case_analysis_result = {
                "prompt_id": prompt_id,
                "src_actual_speedup": src_actual_speedup_this_case,
                "target_estimated_speedups": target_estimated_speedups_this_case
            }
            categorized_analysis_results[lang_category].append(case_analysis_result)

    # --- Calculate Average Speedups Per Language Category ---
    categorized_avg_speedups = {}
    for lang_cat, acc in categorized_speedup_accumulators.items():
        src_avg = sum(acc['src_actual']) / len(acc['src_actual']) if acc['src_actual'] else 0
        target_avg_by_model = {}
        for target_key, speeds in acc['target_estimated'].items():
            target_avg_by_model[target_key] = sum(speeds) / len(speeds) if speeds else 0

        categorized_avg_speedups[lang_cat] = {
            "src_actual_avg": src_avg,
            "target_estimated_avg": target_avg_by_model
        }

    # --- Calculate Overall Average Speedup ---
    final_overall_src_avg = sum(all_src_actual_speedups) / len(all_src_actual_speedups) if all_src_actual_speedups else 0
    final_overall_target_avg_by_model = {}
    for target_key, speeds in all_target_estimated_speedups_by_model.items():
        final_overall_target_avg_by_model[target_key] = sum(speeds) / len(speeds) if speeds else 0

    return {
        "categorized_case_results": categorized_analysis_results,
        "categorized_avg_speedups": categorized_avg_speedups,
        "overall_src_actual_avg_speedup": final_overall_src_avg,
        "overall_target_estimated_avg_speedups": final_overall_target_avg_by_model,
        "src_model_key": src_model_key,
        "target_model_keys": target_model_keys
    }

# --- Plotting Function ---
def plot_speedup_results_combined(all_analysis_results: dict):
    """
    Plots speedup analysis results for multiple source models on a single figure with subplots.
    Each subplot shows one source model's actual speedup and other models' estimated speedup.
    """
    # Get all unique model keys across all analyses for consistent coloring
    all_unique_model_keys = sorted(list(TOKENIZER_MAP.keys()))

    # Assign a consistent color to each model key
    colors = plt.cm.get_cmap('tab10', len(all_unique_model_keys))
    model_color_map = {key: colors(i) for i, key in enumerate(all_unique_model_keys)}

    # Define all categories for X-axis (language categories + overall average)
    categories_keys = LANG_CATEGORIES_KEYS + [OVERALL_CATEGORY_KEY]
    category_display_labels = LANG_CATEGORIES_DISPLAY_LABELS + [OVERALL_CATEGORY_DISPLAY_LABEL]

    num_src_models = len(ANALYSIS_MODELS) # Number of subplots
    fig, axes = plt.subplots(1, num_src_models, figsize=(6.5 * num_src_models, 9.5), sharey=True)
    if num_src_models == 1:
        axes = [axes] # Ensure axes is iterable even for a single subplot

    fig.suptitle('EAGLE-3 Compression Ratio', fontsize=16, y=0.98)

    # Iterate through each source model's analysis results and plot on its respective subplot
    for ax_idx, src_model_key_iter in enumerate(ANALYSIS_MODELS):
        ax = axes[ax_idx]
        analysis_data = all_analysis_results[src_model_key_iter]

        current_src_display_name = get_model_display_name(src_model_key_iter)

        # Get all models that will be plotted in this specific subplot (src + its targets)
        current_subplot_model_keys = [analysis_data['src_model_key']] + analysis_data['target_model_keys']
        current_subplot_model_display_names = [get_model_display_name(key) for key in current_subplot_model_keys]

        bar_width = 0.20 # Increased bar width
        num_bars_per_group = len(current_subplot_model_display_names)

        x_category_centers = np.arange(len(categories_keys))

        # Iterate through each model (src and its targets) to plot its bars across categories
        for i, model_display_name in enumerate(current_subplot_model_display_names):
            y_values_for_model = []

            for lang_cat_key in categories_keys:
                if lang_cat_key == OVERALL_CATEGORY_KEY:
                    if model_display_name == current_src_display_name:
                        y_values_for_model.append(analysis_data['overall_src_actual_avg_speedup'])
                    else:
                        y_values_for_model.append(
                            analysis_data['overall_target_estimated_avg_speedups'].get(model_display_name, 0)
                        )
                else: # Specific language categories (zh, en, multi)
                    if lang_cat_key in analysis_data['categorized_avg_speedups']:
                        if model_display_name == current_src_display_name:
                            y_values_for_model.append(analysis_data['categorized_avg_speedups'][lang_cat_key]['src_actual_avg'])
                        else:
                            y_values_for_model.append(
                                analysis_data['categorized_avg_speedups'][lang_cat_key]['target_estimated_avg'].get(model_display_name, 0)
                            )
                    else: # Category not found in analysis_data, append 0
                        y_values_for_model.append(0)

            offset = (i - (num_bars_per_group - 1) / 2) * bar_width

            # Plot bars and provide label for THIS SUBPLOT's legend
            # Add hatch for the SRC (actual) model's bars
            hatch = '////' if model_display_name == current_src_display_name else None
            bars = ax.bar(x_category_centers + offset, y_values_for_model, bar_width,
                          label=model_display_name, # <-- ADDED label here for ax.legend()
                          color=model_color_map[model_display_name],
                          hatch=hatch, edgecolor='black') # Added edgecolor for better visibility of hatch

            # Add bar values on top
            for bar in bars:
                yval = bar.get_height()
                if yval > 0.01: # Only display non-zero values to avoid clutter
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.08, f'{yval:.2f}', # Adjusted text offset
                            ha='center', va='bottom', fontsize=7)

        ax.legend(title="Model", loc='upper left', fontsize=9)

        # Removed ax.set_xlabel() as requested
        if ax_idx == 0: # Only first subplot gets Y-axis label
            ax.set_ylabel('Compression Ratio (x)', fontsize=12)

        # Set subplot title to SRC=xxx
        # ax.set_title(f'TEST ON {current_src_display_name}', fontsize=14)

        ax.set_title(f'Actual of {current_src_display_name} vs. Estimated for Others', fontsize=14)

        ax.set_xticks(x_category_centers)
        ax.set_xticklabels(category_display_labels, fontsize=10, rotation=30, ha='right')

        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjusted rect to fit within the figure without external legend
    plt.savefig('./res/eagle3_compression_ratio.png', dpi=300)
    plt.show()


# --- Running Analysis ---
if __name__ == "__main__":
    print("\n--- Starting analysis for all source models ---")
    all_src_analysis_results = {}

    for src_key in ANALYSIS_MODELS:
        print(f"\n--- Analyzing results for SRC={get_model_display_name(src_key)} ---")
        results_file_for_src = os.path.join('./data', f'{src_key}.json')

        analysis_data_for_this_src = analyze_single_src_model_results(
            results_file=results_file_for_src,
            src_model_key=src_key,
            all_model_keys=list(TOKENIZER_MAP.keys())
        )
        if analysis_data_for_this_src:
            all_src_analysis_results[src_key] = analysis_data_for_this_src
        else:
            print(f"Warning: Analysis for SRC={src_key} failed, skipping its plot.")

    if all_src_analysis_results:
        print("\nGenerating multi-subplot chart...")
        plot_speedup_results_combined(all_src_analysis_results)

    else:
        print("All model analysis failed, unable to generate chart.")

    print("\nAnalysis and plotting complete.")