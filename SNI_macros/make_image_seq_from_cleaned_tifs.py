from ij import IJ, ImagePlus
import os


def process_cleaned_tiffs(cleaned_folder, mask_folder, utrack_folder, mask_index_formula=lambda x: x * 2):
    """
    Process cleaned TIFF images by applying corresponding masks and saving as an image sequence.

    Parameters:
    - cleaned_folder: Path to folder containing cleaned TIFFs.
    - mask_folder: Path to folder containing mask TIFFs.
    - utrack_folder: Destination folder for uTrack processing.
    - mask_index_formula: Function to compute mask indices (default: x * 2).
    """
    # Ensure the uTrack folder exists
    if not os.path.exists(utrack_folder):
        os.makedirs(utrack_folder)

    # Get all cleaned TIFFs
    files = [f for f in os.listdir(cleaned_folder) if f.endswith('_cleaned.tif')]

    # Define the masked movies directory inside the parent folder of cleaned_folder
    parent_folder = os.path.dirname(cleaned_folder)  # Get parent directory
    masked_folder = os.path.join(parent_folder, "masked_movies")

    # Ensure the masked_movies directory exists
    if not os.path.exists(masked_folder):
        os.makedirs(masked_folder)

    for idx, filename in enumerate(files, 1):
        cleaned_path = os.path.join(cleaned_folder, filename)
        imp = IJ.openImage(cleaned_path)

        if imp:
            print("Processing cleaned file: {}".format(cleaned_path))

            # Extract movie name (without extension)
            movie_name = filename.replace('_cleaned.tif', '')

            # Calculate mask index
            # Extract the movie index from the filename (last 3-digit number after "_")
            movie_index = filename.replace('_cleaned.tif', '').split('_')[-1]  # Get last part before _cleaned.tif
            movie_index = int(movie_index)  # Convert to integer

            # Calculate the mask index correctly
            mask_index = mask_index_formula(movie_index)
            mask_index_str = "{:03d}".format(mask_index)  # Zero-pad to 3 digits
            mask_filename = '_'.join(movie_name.split('_')[:-1] + [mask_index_str + '.tif'])
            mask_path = os.path.join(mask_folder, mask_filename)

            # Open the corresponding mask
            mask_imp = IJ.openImage(mask_path)

            if mask_imp:
                for i in range(1, imp.getStackSize() + 1):
                    imp.setSlice(i)
                    mask_imp.setSlice(i)
                    ip_original = imp.getProcessor()
                    ip_mask = mask_imp.getProcessor()

                    # Set pixels in the original image to black where the mask is black
                    for x in range(ip_mask.getWidth()):
                        for y in range(ip_mask.getHeight()):
                            if ip_mask.getPixel(x, y) == 0:  # If mask pixel is black
                                ip_original.putPixel(x, y, 0)  # Set to black

                    imp.updateAndDraw()
            else:
                print(mask_path)
                print("mask_imp not found. exiting")
                break

            # Save the masked image for testing
            masked_tiff_path = os.path.join(masked_folder, "{}_masked.tif".format(movie_name))
            IJ.saveAsTiff(imp, masked_tiff_path)
            print("Saved masked image for testing: {}".format(masked_tiff_path))

            # Create folder named after the movie (without extension) in uTrack
            movie_folder = os.path.join(utrack_folder, movie_name)
            if not os.path.exists(movie_folder):
                os.makedirs(movie_folder)
            # Re-open the saved masked TIFF to ensure it includes all modifications
            imp.close()  # Close current imp before reloading
            imp = IJ.openImage(masked_tiff_path)

            # Ensure the image is active before exporting
            imp.show()
            IJ.selectWindow(imp.getTitle())
            # Save the processed image as an image sequence for uTrack
            # Save as image sequence
            IJ.run(imp, "Image Sequence... ",
                   "select=[{}] dir=[{}] format=TIFF name=image start=1".format(movie_folder, movie_folder))

            print("Processed and saved: {} into {}, applied mask {}".format(filename, movie_folder, mask_filename))

        else:
            print("Could not open: {}".format(filename))
        imp.close()

    print("Processing complete.")

# Path
cleaned_folder1 = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI2\raw_data\2025-07-11_HeLaS3_H3-2-Halo_c25_ActD_peri\Movies\ActD\cleaned_movies"
cleaned_folder2 = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI2\raw_data\2025-07-11_HeLaS3_H3-2-Halo_c25_ActD_peri\Movies\DMSO\cleaned_movies"

mask_folder1 = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI2\raw_data\2025-07-11_HeLaS3_H3-2-Halo_c25_ActD_peri\masks\ActD"
mask_folder2 = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI2\raw_data\2025-07-11_HeLaS3_H3-2-Halo_c25_ActD_peri\masks\DMSO"

utrack_folder1 = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI2\all_data_analysis\for_utrack\2025-07-11_Jason"
utrack_folder2 = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI2\all_data_analysis\for_utrack\2025-06-11_Jason"


mask_index_formula1 = lambda x: (x * 2) # Adjust as needed
mask_index_formula2 = lambda x: (x + 1) 
# Repli-Histo
 
#from cleaned movie to mask math


# Run the function
process_cleaned_tiffs(cleaned_folder1, mask_folder1, utrack_folder1, mask_index_formula1)
process_cleaned_tiffs(cleaned_folder2, mask_folder2, utrack_folder1, mask_index_formula1)
