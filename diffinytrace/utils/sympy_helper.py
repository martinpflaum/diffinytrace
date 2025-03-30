import sympy
from sympy.utilities.lambdify import lambdify

#TODO remove this file from here!!!


def sympy_calc_grad(expr_dict,input_list,return_latex=False):
    
    if not type(expr_dict)==dict:
        raise RuntimeError("expr_dict needs to be a dict! Example: {func: x**2}")
    if len(expr_dict.keys())==1:
        expr_dict_keys = [elem for elem in expr_dict.keys()]
        if ";" in expr_dict_keys[0]:
           pass
        else:
            new_expr_dict = {}
            new_expr_dict[expr_dict_keys[0]+";"] =expr_dict[expr_dict_keys[0]] 
            expr_dict = new_expr_dict
    out_sym = {}
    out_tex = {}
    for key in expr_dict.keys():
        expr = expr_dict[key]
        for input in input_list:
            tmp = sympy.diff(expr,input)
            new_key = "d"+key+"d"+str(input)
            new_tex_func = "\\frac{d"+key[:-1]+"}{d"+sympy.latex(input)+"}"#+"\\label{"+new_key+"}"
            frac_upper = new_key.split(";")[0]
            
            if key[-1] != ";":
                post = new_key.split(";")[1]
                frac_lower_sym = [post.count(str(input))*("d"+str(input)) for input in input_list]
                new_key = frac_upper+";"+''.join(frac_lower_sym)
                    
                
                frac_lower_tex = ""

                for input in input_list:
                    if post.count(str(input)) == 1:
                        frac_lower_tex += f"d"+sympy.latex(input)
                    if post.count(str(input)) > 1:
                        frac_lower_tex += f"d"+sympy.latex(input)+"^{"+str(post.count(str(input)))+"}"
                frac_upper_tex = f'd^{frac_upper.count("d")}{frac_upper[frac_upper.count("d"):]}'
                new_tex_func = "\\frac{"+frac_upper_tex+"}{"+frac_lower_tex+"}"#+"\\label{"+new_key+"}"
                
            out_sym[new_key] = tmp
            out_tex[new_key] = new_tex_func+"="+sympy.latex(tmp)
    
    if return_latex:
        return out_sym,out_tex
    else:
        return out_sym
def sympy_print_align_block(tex_elems):
    out = "\\begin{align}"
    for key in tex_elems.keys():
        out += tex_elems[key]+"\\\\"
    out = out.replace("=","&=")
    out += "\\end{align}"
    print(out)


"""

import sympy as sp
from sympy.utilities.lambdify import lambdastr
# Step 1: Define the SymPy expression
x, y = sp.symbols('x y')
expr = x**2 + y**2

# Step 2: Use lambdastr to get the string representation

symbols = (x, y)
def create_function_from_str(expr_string):
    def func(*args, **kwargs):
        # Create a local scope dictionary for variables
        local_scope = {f'arg{i}': arg for i, arg in enumerate(args)}
        local_scope.update(kwargs)
        # Evaluate the expression using the local scope
        return eval(expr_string, {}, local_scope)
    return func

def create_function_from_sympy(symbols, expr):
    function_str = lambdastr(symbols, expr)
    expr_string = function_str.split(":")[1]
    
    # Create the function from the string
    f = create_function_from_str(expr_string)
    return f
"""