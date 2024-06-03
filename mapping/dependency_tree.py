import ast
import json
import sys

ignore_funcs = [
    'get_logging_level', 'audit_logger', 'decorator',
    'collect_dependencies', 'decorator', 'wrapper', 
    'configure_logger'
]

def analyze_dependencies(source_code):
    tree = ast.parse(source_code)
    
    class FunctionDefVisitor(ast.NodeVisitor):
        def __init__(self):
            self.functions = {}

        def visit_FunctionDef(self, node):
            if node.name not in ignore_funcs:
                self.functions[node.name] = node
            self.generic_visit(node)

    visitor = FunctionDefVisitor()
    visitor.visit(tree)
    
    dependencies = {name: {"downstream": [], "upstream": []} for name in visitor.functions.keys()}
    
    class FunctionCallVisitor(ast.NodeVisitor):
        def __init__(self, dependencies):
            self.dependencies = dependencies
            self.current_function = None

        def visit_FunctionDef(self, node):
            self.current_function = node.name
            if self.current_function in self.dependencies:
                self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                function_called = node.func.id
                if function_called not in ignore_funcs:
                    if self.current_function and function_called in self.dependencies:
                        self.dependencies[self.current_function]["downstream"].append(function_called)
                        self.dependencies[function_called]["upstream"].append(self.current_function)
            self.generic_visit(node)
    
    for name, node in visitor.functions.items():
        visitor = FunctionCallVisitor(dependencies)
        visitor.visit(node)
    
    # Deduplicate dependencies
    for key in dependencies:
        dependencies[key]["downstream"] = list(set(dependencies[key]["downstream"]))
        dependencies[key]["upstream"] = list(set(dependencies[key]["upstream"]))
    
    return dependencies

def save_dependencies_as_json(dependencies, output_filename):
    with open(output_filename, 'w') as file:
        json.dump(dependencies, file, indent=4)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python dependency_tree.py <module_filename> <output_filename>")
    else:
        module_filename = sys.argv[1]
        output_filename = sys.argv[2]
        print(f"Input: {module_filename}")
        print(f"Output: {output_filename}")
        with open(module_filename, 'r') as file:
            source_code = file.read()

        dependencies = analyze_dependencies(source_code)
        save_dependencies_as_json(dependencies, output_filename)
