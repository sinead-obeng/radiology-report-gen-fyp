import os
import sys
import numpy as np
from transformers import BioGptTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from rouge import Rouge

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import REPORTS_DIR, REPORTS_TEST_DIR

# Load BioGPT tokenizer
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
tokenizer.pad_token = tokenizer.eos_token
rouge = Rouge()


def tokenize_generated_reports(generated_reports_dir):
    """Tokenize generated reports and save them in the same directory as .npy files."""
    for root, _, files in os.walk(generated_reports_dir):
        for file in files:
            if file.endswith(".txt"):
                report_file_path = os.path.join(root, file)
                
                # Read the generated report
                with open(report_file_path, 'r', encoding="utf-8") as f:
                    generated_report = f.read().strip()

                # Tokenize the report with padding, truncation, and max_length
                tokens = tokenizer(
                    generated_report, 
                    padding=False, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                )

                # Save the tokenized report as a .npy file (input_ids and attention_mask)
                tokenized_report_path = report_file_path.replace('.txt', '_tokens.npy')
                np.save(tokenized_report_path, tokens['input_ids'].numpy())

                print(f"Tokenized report saved: {tokenized_report_path}")


def flatten(lst):
    """Flatten a nested list into a 1D list."""
    return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]


def remove_padding(tokens, pad_token_id=1):
    return [token for token in tokens if token != pad_token_id]


def calculate_bleu_score(original_tokens, generated_tokens):
    """
    Calculate the BLEU score between original and generated reports.
    Uses smoothing to handle short reports.
    """

    cleaned_original_tokens = remove_padding(original_tokens[0]) 
    cleaned_generated_tokens = remove_padding(generated_tokens)

    cleaned_original_tokens = flatten(cleaned_original_tokens)
    cleaned_generated_tokens = flatten(cleaned_generated_tokens)

    # print("Cleaned Original Tokens:", cleaned_original_tokens)
    # print("Cleaned Generated Tokens:", cleaned_generated_tokens)

    smoothie = SmoothingFunction().method4
    return sentence_bleu([cleaned_original_tokens], cleaned_generated_tokens, smoothing_function=smoothie)


def calculate_rouge_score(original_text, generated_text):
    """
    Calculate ROUGE score between original and generated reports.
    """
    rouge_scores = rouge.get_scores(generated_text, original_text)
    
    # Print the full ROUGE scores to check its structure
    # print("ROUGE scores:", rouge_scores)
    
    # Check if 'rouge-1' exists in the output before trying to access it
    if 'rouge-1' in rouge_scores[0]:
        rouge_1_f1 = rouge_scores[0]['rouge-1']['f']
    else:
        rouge_1_f1 = 0

    if 'rouge-2' in rouge_scores[0]:
        rouge_2_f1 = rouge_scores[0]['rouge-2']['f']
    else:
        rouge_2_f1 = 0

    if 'rouge-l' in rouge_scores[0]:
        rouge_l_f1 = rouge_scores[0]['rouge-l']['f']
    else:
        rouge_l_f1 = 0

    return rouge_1_f1, rouge_2_f1, rouge_l_f1


def calculate_bert_score(original_text, generated_text):
    """
    Calculate BERTScore for semantic similarity using a BERT-like model.
    """
    # Ensure text is stripped and cleaned of special tokens
    original_text = original_text.replace("<pad>", "").strip()
    generated_text = generated_text.replace("<pad>", "").strip()

    # Debugging outputs to verify cleaned inputs
    # print(f"Original: {original_text}")
    # print(f"Generated: {generated_text}")

    # BERTScore requires lists of text samples
    original_texts = [original_text]
    generated_texts = [generated_text]

    # Ensure matching pairs
    assert len(generated_texts) == len(original_texts), "Mismatch in generated and reference reports."

    # Calculate BERTScore using a supported model
    P, R, F1 = score(
        generated_texts,
        original_texts,
        lang="en",
        model_type="bert-base-uncased",
        rescale_with_baseline=True 
    )

    # Return mean Precision, Recall, and F1 scores
    return P.mean().item(), R.mean().item(), F1.mean().item()


def evaluate_metrics(generated_reports_dir, original_reports_dir):
    """
    Evaluate BLEU, ROUGE, and BERTScore for all generated reports by comparing them with original reports.
    Outputs the average scores.
    """
    total_bleu = 0
    total_rouge_1_f1 = 0
    total_rouge_2_f1 = 0
    total_rouge_l_f1 = 0
    total_bert_f1 = 0
    report_count = 0

    for root, _, files in os.walk(generated_reports_dir):
        for file in files:
            if file.endswith("_tokens.npy"):
                generated_path = os.path.join(root, file)

                # Find corresponding original report path
                relative_path = os.path.relpath(generated_path, generated_reports_dir)
                original_path = os.path.join(original_reports_dir, relative_path)

                if not os.path.exists(original_path):
                    continue

                # Load tokenized reports
                generated_tokens = np.load(generated_path)
                original_tokens = np.load(original_path)

                # Decode tokenized reports into text
                generated_report_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                original_report_text = tokenizer.decode(original_tokens[0], skip_special_tokens=True)

                # Calculate BLEU score
                bleu_score = calculate_bleu_score(original_tokens.tolist(), generated_tokens.tolist())
                total_bleu += bleu_score

                # Calculate ROUGE score
                rouge_1_f1, rouge_2_f1, rouge_l_f1 = calculate_rouge_score(original_report_text, generated_report_text)
                total_rouge_1_f1 += rouge_1_f1
                total_rouge_2_f1 += rouge_2_f1
                total_rouge_l_f1 += rouge_l_f1

                # Calculate BERTScore
                bert_p, bert_r, bert_f1 = calculate_bert_score(original_report_text, generated_report_text)
                total_bert_f1 += bert_f1

                report_count += 1

    # Compute and print the average scores
    average_bleu = total_bleu / report_count if report_count > 0 else 0
    average_rouge_1_f1 = total_rouge_1_f1 / report_count if report_count > 0 else 0
    average_rouge_2_f1 = total_rouge_2_f1 / report_count if report_count > 0 else 0
    average_rouge_l_f1 = total_rouge_l_f1 / report_count if report_count > 0 else 0
    average_bert_f1 = total_bert_f1 / report_count if report_count > 0 else 0

    print(f"\nAverage BLEU Score: {average_bleu:.4f}")
    print(f"Average ROUGE-1 F1: {average_rouge_1_f1:.4f}")
    print(f"Average ROUGE-2 F1: {average_rouge_2_f1:.4f}")
    print(f"Average ROUGE-L F1: {average_rouge_l_f1:.4f}")
    print(f"Average BERTScore - Precision: {bert_p:.4f}, Recall: {bert_r:.4f}, F1: {average_bert_f1:.4f}")



if __name__ == "__main__":
    generated_reports_dir = os.path.join(REPORTS_DIR, "test") 
    # Uncomment if tokenization is needed before evaluation
    print("Tokenizing generated reports...")
    tokenize_generated_reports(generated_reports_dir)

    # Evaluate BLEU score
    print("\nEvaluating BLEU, ROUGE and BERT scores...")
    evaluate_metrics(generated_reports_dir, REPORTS_TEST_DIR)
