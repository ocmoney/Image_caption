from datasets import load_dataset
import numpy as np

def analyze_caption_lengths():
    # Load the dataset
    dataset = load_dataset("nlphuji/flickr30k")
    
    # Get all captions from the test split (which contains all data)
    # Each image has multiple captions, so we flatten them
    captions = [cap for caps in dataset['test']['caption'] for cap in caps]
    
    # Calculate lengths
    lengths = [len(caption.split()) for caption in captions]
    
    # Calculate statistics
    max_length = max(lengths)
    min_length = min(lengths)
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    std_length = np.std(lengths)
    p5_length = np.percentile(lengths, 5)
    p10_length = np.percentile(lengths, 10)
    p90_length = np.percentile(lengths, 90)
    p95_length = np.percentile(lengths, 95)
    p96_length = np.percentile(lengths, 96)
    p97_length = np.percentile(lengths, 97)
    p98_length = np.percentile(lengths, 98)
    p99_length = np.percentile(lengths, 99)
    p100_length = np.percentile(lengths, 100)  # Should match max_length
    
    # Print results
    print("\nCaption Length Statistics:")
    print(f"Maximum length: {max_length} words")
    print(f"Minimum length: {min_length} words")
    print(f"Mean length: {mean_length:.2f} words")
    print(f"Median length: {median_length} words")
    print(f"Standard deviation: {std_length:.2f} words")
    print(f"5th percentile: {p5_length:.2f} words")
    print(f"10th percentile: {p10_length:.2f} words")
    print(f"90th percentile: {p90_length:.2f} words")
    print(f"95th percentile: {p95_length:.2f} words")
    print(f"96th percentile: {p96_length:.2f} words")
    print(f"97th percentile: {p97_length:.2f} words")
    print(f"98th percentile: {p98_length:.2f} words")
    print(f"99th percentile: {p99_length:.2f} words")
    print(f"100th percentile: {p100_length:.2f} words")
    
    # Print some examples
    print("\nExample Captions:")
    print(f"Shortest caption ({min_length} words):")
    print(f"- {captions[lengths.index(min_length)]}")
    print(f"\nLongest caption ({max_length} words):")
    print(f"- {captions[lengths.index(max_length)]}")
    
    return {
        'max_length': max_length,
        'min_length': min_length,
        'mean_length': mean_length,
        'median_length': median_length,
        'std_length': std_length,
        'p5_length': p5_length,
        'p10_length': p10_length,
        'p90_length': p90_length,
        'p95_length': p95_length,
        'p96_length': p96_length,
        'p97_length': p97_length,
        'p98_length': p98_length,
        'p99_length': p99_length,
        'p100_length': p100_length
    }

if __name__ == "__main__":
    stats = analyze_caption_lengths() 