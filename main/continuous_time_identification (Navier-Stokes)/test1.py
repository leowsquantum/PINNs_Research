# Text you want to write to the file
text_to_save = "Hello, this is the text I want to save in a file."

# Specify the filename
filename = "pinn_ns_results/example.txt"

# Open the file in write mode ('w'). This will create the file if it doesn't exist
# or overwrite it if it does.
with open(filename, 'w') as file:
    file.write(text_to_save)

print(f"File '{filename}' has been saved.")
