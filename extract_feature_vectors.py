import os
import javalang
import pandas as pd
import re

START_DIR = 'resources/defects4j-checkout-closure-1f/src/com/google/javascript/jscomp'
CSV_FILE_NAME = 'feature vector file.csv'


def count_method_invocations(method):
    return len([node for path, node in method if isinstance(node, javalang.tree.MethodInvocation)])


def count_statements(method):
    return len([node for path, node in method if isinstance(node, javalang.tree.Statement)])


def count_complexity(method):
    return len([node for path, node in method if isinstance(node, (
        javalang.tree.IfStatement, javalang.tree.WhileStatement, javalang.tree.ForStatement,
        javalang.tree.DoStatement))])


def count_returns(method):
    return len([node for path, node in method if isinstance(node, javalang.tree.ReturnStatement)])


def count_exceptions(method):
    return len(method.throws) if method.throws else 0


def count_block_comments(java_code):
    return len(re.findall(r'/\*.*?\*/', java_code, re.DOTALL))


def method_name_lengths(methods):
    return [len(method.name) for method in methods]


def count_words_in_block_comments(java_code):
    comments = re.findall(r'/\*.*?\*/', java_code, re.DOTALL)
    return sum(len(re.findall(r'\w+', comment)) for comment in comments)


def comments_per_statement(java_code, num_statements):
    words_in_comments = count_words_in_block_comments(java_code)
    return words_in_comments / num_statements if num_statements > 0 else 0


def check_if_enum_or_interface(tree, filepath):
    filename = os.path.basename(filepath).replace('java', '')
    for path, node in tree:
        if isinstance(node, javalang.tree.InterfaceDeclaration) or isinstance(node,
                                                                              javalang.tree.EnumDeclaration) and node.name == filename:
            return True
    return False


def get_class_attrib(tree, java_code, filename):
    package_name = ""

    class_metrics = {
        'class': None,
        'MTH': 0,
        'FLD': 0,
        'RFC': 0,
        'INT': 0,
        'SZ': 0,
        'CPX': 0,
        'EX': 0,
        'RET': 0,
        'BCM': 0,
        'NML': 0,
        'WRD': 0,
        'DCM': 0
    }

    methods = []
    num_statements = 0

    filename = os.path.basename(filename).replace('.java', '')

    for path, node in tree:
        if isinstance(node, javalang.tree.PackageDeclaration):
            package_name = node.name

        if isinstance(node, javalang.tree.ClassDeclaration):
            class_name = node.name
            if class_name != filename:  # We do not care about inner classes.
                continue

            if package_name:
                class_metrics['class'] = f"{package_name}.{class_name}"
            else:
                class_metrics['class'] = class_name
            class_metrics['INT'] = len(node.implements) if node.implements else 0 # Want to count the number of implements

        if isinstance(node, javalang.tree.FieldDeclaration):
            class_metrics['FLD'] += len(node.declarators)   # Accumulating the fields

        if isinstance(node, javalang.tree.MethodDeclaration):
            class_metrics['MTH'] += 1
            methods.append(node)
            num_statements += count_statements(node)
            class_metrics['RFC'] += 1 + count_method_invocations(node)
            class_metrics['SZ'] = max(class_metrics['SZ'], count_statements(node))    # Gets max over all methods
            class_metrics['CPX'] = max(class_metrics['CPX'], count_complexity(node))
            class_metrics['EX'] = max(class_metrics['EX'], count_exceptions(node))
            class_metrics['RET'] = max(class_metrics['RET'], count_returns(node))

    if methods:
        class_metrics['NML'] = sum(method_name_lengths(methods)) / len(methods)

    class_metrics['BCM'] = count_block_comments(java_code)
    class_metrics['WRD'] = count_words_in_block_comments(java_code)
    class_metrics['DCM'] = comments_per_statement(java_code, num_statements)

    return class_metrics


def get_files(start_dir: str) -> list[str]:
    """
    Lists the files in a recursive manner starting from a particular directory.

    :param start_dir: The path of the directory on which we want to recurse on to list the files.
    :return: The list of files.
    """
    files = list()
    for dp, di, fn in os.walk(start_dir):
        files.extend([os.path.join(dp, f) for f in fn])
        for dirs in di:
            files.extend(get_files(dirs))
    return files


def filter_file_by_extension(files, extension='java'):
    """
     Filters a list of filepaths which have the specified extension.

    :param extension: Extension that a file is supposed to have. E.g. Java, Python, etc.
    :param files: The list of filepaths that we want to filter out.
    :return: List of filepaths with a set.
    """
    return list(filter(lambda filename: filename.endswith(extension), files))


def get_and_parse_tree(file_list):
    metrics = list()
    for i in file_list:
        with open(i, 'r') as f:
            data = f.read()
        tree = javalang.parse.parse(data)
        # if not check_if_enum_or_interface(tree, i):
        class_metric = get_class_attrib(tree, data, i)
        if class_metric['class'] is not None:
            metrics.append(class_metric)
    return pd.DataFrame(metrics)


def save_to_csv(df, filename):
    df.to_csv(filename)


if __name__ == '__main__':
    files = get_files(START_DIR)
    files = filter_file_by_extension(files)
    feature_vectors = get_and_parse_tree(files)
    save_to_csv(feature_vectors, CSV_FILE_NAME)
