import sys

input_file = "embeddings/glove.840B.300d.txt"
output_file = "embeddings/glove.840B.300d.cleaned.txt"
embedding_dim = 300

print(f"Cleaning {input_file}...")
valid_lines = 0
skipped_lines = 0

with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin:
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                parts = line.strip().split()
                if len(parts) != embedding_dim + 1:
                    skipped_lines += 1
                    continue
                
                word = parts[0]
                vector = [float(x) for x in parts[1:]]
                
                fout.write(line)
                valid_lines += 1
                
                if line_num % 100000 == 0:
                    print(f"Processed {line_num} lines... ({valid_lines} valid, {skipped_lines} skipped)")
                    
            except (ValueError, IndexError):
                skipped_lines += 1
                continue

print(f"\n Valid lines: {valid_lines}, Skipped: {skipped_lines}")
print(f"Cleaned file saved to: {output_file}")
print(f"\nTo use the cleaned file, run:")
print(f"  mv {output_file} {input_file}")
