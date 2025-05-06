from datasets import load_dataset


dataset = load_dataset("nlphuji/flickr30k", split="test")

caption_lengths = []
for i in range(len(dataset)):
    for caption in dataset[i]["caption"]:
        caption_lengths.append(len(caption.split(" ")))

sorted_caption_lengths = sorted(caption_lengths)
print("Average caption length: ", sum(sorted_caption_lengths) / len(sorted_caption_lengths))
print("Max caption length: ", sorted_caption_lengths[-1])
print("Median caption length: ", sorted_caption_lengths[len(sorted_caption_lengths) // 2])
print("p90 caption length: ", sorted_caption_lengths[int(len(sorted_caption_lengths) * 0.9)])
