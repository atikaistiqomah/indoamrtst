import penman
import re
from typing import Callable

import penman.layout
node_name_matcher = re.compile(r"^[a-z]{1,3}([0-9]+)?$")

def is_node_name(current_token):
    return node_name_matcher.match(current_token) is not None

def make_no_metadata_graph(g: penman.Graph):
    return penman.Graph(
        triples=g.triples,
        top=g.top,
        epidata=g.epidata,
        metadata={}
    )

def old_to_amr_with_pointer(amr: str):
    result = ""
    status = "find_first_left"
    level = 0
    node_name_to_pointer_map: dict[str, str] = {}
    unresolved_node_names = set()
    next_pointer_id = 0
    current_token = ""
    for c in amr:
        if status == "find_first_left":
            if c == "(":
                result += "( "
                level += 1
                status = "find_begin_of_new_node_name"
            # else: ignore
                
        elif status == "find_begin_of_new_node_name":
            if c in "abcdefghijklmnopqrstuvwxyz":
                current_token = c
                status = "find_end_of_new_node_name"
            elif not c.isspace():
                raise ValueError(f"Unexpected begin of node name: \"{c}\"")
            # else: c is a space; ignore

        elif status == "find_end_of_new_node_name":
            if c in "abcdefghijklmnopqrstuvwxyz-0123456789":
                current_token += c
            elif c.isspace() or c == "/":
                if is_node_name(current_token):
                    node_name = current_token
                    if node_name in node_name_to_pointer_map:
                        if node_name in unresolved_node_names:
                            pointer = node_name_to_pointer_map[node_name]
                            unresolved_node_names.remove(node_name)
                        else:
                            raise ValueError(f"Duplicate node name: {node_name}")
                    else:
                        pointer = f"<pointer:{next_pointer_id}>"
                        next_pointer_id += 1
                        node_name_to_pointer_map[node_name] = pointer
                    
                    result += f"{pointer} "

                    if c != "/":
                        status = "find_slash"
                    else:
                        status = "find_begin_of_concept"
                else:
                    raise ValueError(f"Unexpected node name: \"{current_token}\"")

            else:
                raise ValueError(f"Unexpected char of node name: \"{c}\"")
            
        elif status == "find_slash":
            if c == "/":
                status = "find_begin_of_concept"

            elif not c.isspace():
                raise ValueError(f"Expecting slash, got \"{c}\"")
            # else: ignore

        elif status == "find_begin_of_concept":
            if c in "abcdefghijklmnopqrstuvwxyz":
                current_token = c
                status = "find_end_of_concept"
            elif not c.isspace():
                raise ValueError(f"Unexpected begin of concept: \"{c}\"")
            # else: c is a space; ignore

        elif status == "find_end_of_concept":
            if c in "abcdefghijklmnopqrstuvwxyz-0123456789":
                current_token += c
            elif c.isspace() or c == ")":
                result += f"{current_token}"
                if c != ")":
                    status = "find_right_or_begin_of_relation"
                else:
                    level -= 1
                    result += " )"
                    if level == 0:
                        status = "end"
                    else:
                        status = "find_right_or_begin_of_relation"

            else:
                raise ValueError(f"Unexpected char of concept: \"{c}\"")

        elif status == "find_right_or_begin_of_relation":
            if c == ")":
                level -= 1
                result += " )"
                if level == 0:
                    status = "end"
                # else: keep the status
            
            elif c == ":":
                current_token = c
                status = "find_end_of_relation"
        
            elif not c.isspace():
                raise ValueError(f"Expecting right parenthesis or begin of relation, got \"{c}\"")
            
            # else: c is space; ignore
        
        elif status == "find_end_of_relation":
            if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789":
                current_token += c
            elif c.isspace() or c == "(" or c == "\"":
                result += f" {current_token} "
                if c == "(":
                    result += "( "
                    level += 1
                    status = "find_begin_of_new_node_name"
                elif c == "\"":
                    result += c
                    status = "find_end_of_literal_value"
                else:
                    status = "find_left_or_begin_of_value"
            else:
                raise ValueError(f"Unexpected char of relation: \"{c}\"")
            
        elif status == "find_left_or_begin_of_value":
            if c == "(":
                result += "( "
                level += 1
                status = "find_begin_of_new_node_name"
            
            elif c in "abcdefghijklmnopqrstuvwxyz+-0123456789":
                # It can be a node name or non-literal constant.
                current_token = c
                status = "find_end_of_non_literal_value"

            elif c == "\"":
                result += c
                status = "find_end_of_literal_value"

            elif not c.isspace():
                raise ValueError(f"Expecting left parenthesis or begin of value, got \"{c}\"")

            # else: ignore
            
        elif status == "find_end_of_non_literal_value":
            # It includes float.
            if c in "abcdefghijklmnopqrstuvwxyz-0123456789.":
                current_token += c
            elif c.isspace() or c == ")":
                if is_node_name(current_token):
                    node_name = current_token
                    if node_name in node_name_to_pointer_map:
                        pointer = node_name_to_pointer_map[node_name]
                    else:
                        pointer = f"<pointer:{next_pointer_id}>"
                        next_pointer_id += 1
                        node_name_to_pointer_map[node_name] = pointer
                        unresolved_node_names.add(node_name)
                    
                    result += f"{pointer}"
                    
                else:
                    result += f"{current_token}"

                if c != ")":
                    status = "find_right_or_begin_of_relation"
                else:
                    level -= 1
                    result += " )"
                    if level == 0:
                        status = "end"
                    else:
                        status = "find_right_or_begin_of_relation"

            else:
                raise ValueError(f"Unexpected char of node name or concept: \"{c}\"")
            
        elif status == "find_end_of_literal_value":
            result += c
            if c == "\"":
                status = "find_right_or_begin_of_relation"

        elif status == "end":
            if not c.isspace():
                raise ValueError(f"Expecting end, got {c}")

        else:
            raise ValueError(f"Unexpected status: {status}")
        
    if status != "end":
        raise ValueError(f"Unexpected end status: {status}")
    
    if len(unresolved_node_names) > 0:
        raise ValueError(f"Unresolved node names: {unresolved_node_names}")
        
    return result

def _is_z_prefix_variable(var: str):
    if len(var) <= 1:
        return False

    if var[0] != "z":
        return False
    
    return var[1:].isdigit()

def _zpv_to_pointer(zp_var: str):
    # zp stands for z-prefix
    return f"<pointer:{zp_var[1:]}>"

def to_amr_with_pointer(amr_str: str):
    g = penman.decode(amr_str)

    # Convert all variables to z-prefix
    var_list = []
    possibly_already_z_prefix = True

    for var, _, _ in g.instances():
        if var not in var_list:
            if not _is_z_prefix_variable(var):
                possibly_already_z_prefix = False
            
            var_list.append(var)

    if possibly_already_z_prefix:
        new_g = _convert_all_variables_to_pointer(g, _is_z_prefix_variable, _zpv_to_pointer)

    else:
        next_pointer_id = 0
        var_to_pointer: dict[str, str] = {}
        for var in var_list:
            pointer = f"<pointer:{next_pointer_id}>"
            var_to_pointer[var] = pointer
            next_pointer_id += 1

        is_variable_fn: Callable[[str], bool]  = lambda x: x in var_to_pointer
        to_pointer_fn: Callable[[str], str] = lambda x: var_to_pointer[x]

        new_g = _convert_all_variables_to_pointer(g, is_variable_fn, to_pointer_fn)

    new_g_encoded = penman.encode(new_g, indent=None)
    
    new_amr_str = ""
    in_literal = False
    prev_is_ignore_char = False
    for c in new_g_encoded:
        if in_literal:
            new_amr_str += c
            if c == "\\":
                prev_is_ignore_char = not prev_is_ignore_char
            else:
                if c == "\"" and (not prev_is_ignore_char):
                    in_literal = False
                prev_is_ignore_char = False
        else:
            if c == "(":
                new_amr_str += "( "
            elif c == ")":
                new_amr_str += " )"
            elif c == "/":
                pass # Ignore this symbol.
            elif c == " ":
                if new_amr_str != "" and new_amr_str[-1] != " ":
                    new_amr_str += " "
            else:
                new_amr_str += c
                if c == "\"":
                    in_literal = True

    return new_amr_str

def _convert_all_variables_to_pointer(
        g: penman.Graph,
        is_variable_fn: Callable[[str], bool],
        to_pointer_fn: Callable[[str], str]):
    new_triples = []
    new_epidata = {}
    for triple in g.triples:
        epidata_value = g.epidata[triple]
        src, rel, tgt = triple
        if is_variable_fn(src):
            src = to_pointer_fn(src)

        if is_variable_fn(tgt):
            tgt = to_pointer_fn(tgt)

        new_single_triple = (src, rel, tgt)
        new_triples.append(new_single_triple)

        new_epidata_value = []
        for evi in epidata_value:
            if isinstance(evi, penman.layout.Push) and is_variable_fn(evi.variable):
                new_epidata_value.append(
                        penman.layout.Push(to_pointer_fn(evi.variable))
                    )
            else:
                new_epidata_value.append(evi)
            
        new_epidata[new_single_triple] = new_epidata_value

    new_top = g.top
    if is_variable_fn(new_top):
        new_top = to_pointer_fn(new_top)
        
    return penman.Graph(
        triples=new_triples,
        top=new_top,
        epidata=new_epidata,
        metadata=g.metadata
    )

def count_node_in_graph(g: penman.Graph):
    nodes = set([g.top])
    for source, role, target in g.triples:
        if role == ":instance":
            continue

        nodes.add(source)
        nodes.add(target)

    return len(nodes)

def count_token_in_text(s: str):
    return len("".join([(c if c.isalnum() else " ") for c in s]).strip().split())
