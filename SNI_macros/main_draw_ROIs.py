from draw_ROIs import InteractiveROIDrawer

if __name__ == '__main__':
    bf_folder = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI_SMI1\raw_data\2025-07-03_RPE1_H3-2-Halo_Doxo_RH\BF_images\Doxo"
    movie_folder = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI_SMI1\raw_data\2025-07-03_RPE1_H3-2-Halo_Doxo_RH\Movies\Doxo\cleaned_movies"
    output_folder = r"C:\Users\jason\OneDrive\Documents\MaeshimaLab\experiments\SNI_SMI1\raw_data\2025-07-03_RPE1_H3-2-Halo_Doxo_RH\masks\Doxo"
    index_formula = "index - 1"  # from bf image to movie

    roi_tool = InteractiveROIDrawer(bf_folder, movie_folder, output_folder, index_formula)
    roi_tool.run()
