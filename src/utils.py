import numpy as np



def load_image_embeddings(file_path):
    # show the first 10 image ids and embeddings
    with np.load('./embeddings/image_embeddings.npz') as f:
        # Print the keys available in the npz file
        print("Keys in the npz file:", f.files)

        # Access and print the first item of each array in the file
        for key in f.files:
            print(f"First item of {key}:", f[key][0])

        # If you specifically want to access image_ids and embeddings:
        if 'image_ids' in f.files and 'embeddings' in f.files:
            print("First image ID:", f['image_ids'][0])
            # check type
            print(type(f['image_ids'][0]))
            print("First embedding:", f['embeddings'][0])
        else:
            print("'image_ids' or 'embeddings' not found in the file")


if __name__ == "__main__":
    load_image_embeddings("./embeddings/image_embeddings.npz")
