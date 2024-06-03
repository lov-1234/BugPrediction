from extract_feature_vectors import get_files, CSV_FILE_NAME, save_to_csv
import pandas as pd

NEW_FEATURE_VECTOR_FILENAME = 'new feature vector file.csv'


def read_and_list_buggy_classes(file_list):
    buggy_files = list()
    for i in file_list:
        with open(i, 'r') as f:
            data = f.readlines()
        for dat in data:
            dat = dat.strip()
            if 'jscomp' in dat.strip():  # We are only reading the files in the jscomp directory and not the rhino
                # one as the modified files also contain the rhino directory java files
                buggy_files.append(dat.strip())
    return buggy_files


def add_buggy_column(df, buggy_files):
    df['buggy'] = df['class'].apply(lambda x: 1 if x in buggy_files else 0)
    return df


def read_feature_csv_file(feature_vector_csv_file_path):
    return pd.read_csv(feature_vector_csv_file_path)


if __name__ == '__main__':
    mod_files = get_files('resources/modified_classes')
    mod_files = read_and_list_buggy_classes(mod_files)
    feature_vectors = read_feature_csv_file(CSV_FILE_NAME)
    feature_vectors = add_buggy_column(feature_vectors, mod_files)
    save_to_csv(feature_vectors, NEW_FEATURE_VECTOR_FILENAME)
