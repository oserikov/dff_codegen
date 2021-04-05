import re
from typing import Dict
from copy import deepcopy
import gdown
from flask import Flask, render_template
from lxml import etree
import uuid

BOILERPLATE_CODE_FNAME = "boilerplate_head.py_"

DIAGRAM_LINK = "https://drive.google.com/uc?export=download&id=11-cXCfyd-7KFkWnuR0OC-31t3j7SbAZ7"
XML_FNAME = "alexa_dream_scripts.xml"


def download_diagrams():
    gdown.download(DIAGRAM_LINK, XML_FNAME, quiet=False)


class Edge:
    def __init__(self, cell):
        self.source = cell.get("source")
        self.target = cell.get("target")


class Node:
    heights_visited = []

    def __init__(self, cell):
        node_head_and_text = cell.get("value", '').split("<br>", maxsplit=1)
        node_head = node_head_and_text[0]
        node_text = node_head_and_text[-1]

        self.value = cell.get("value")
        self.head = node_head
        self.text = node_text
        self.id = cell.get("id")
        self.var = to_upper_snake_case(node_head)
        self._children = []
        self.dfs_step = self.id
        self._height_cache = None
        # self._is_recursive = None

    @property
    def children(self):
        self._children = list(set(self._children))
        return self._children

    def is_usr(self):
        return "user" in self.var.lower() or "usr" in self.var.lower()

    def is_sys(self):
        # return "system" in self.var.lower() or "sys" in self.var.lower()
        return not self.is_usr()

    @property
    def method(self):
        if self.is_usr():
            method = self.var.lower() + "_response"
        else:
            method = self.var.lower() + "_request"
        return method

    def _transition_method(self):
        if self.is_usr():
            if len(self.children) > 1:
                transition_method = "add_user_serial_transitions"
            else:
                transition_method = "add_user_transition"
        else:
            if len(self.children) > 1:
                transition_method = "add_system_serial_transitions"
            else:
                transition_method = "add_system_transition"
        return transition_method

    def _height(self):
        if self._height_cache is not None:
            return self._height_cache
        if self.id in Node.heights_visited:
            # self._is_recursive = True
            return 0
        Node.heights_visited.append(self.id)
        height =  1 + max([c._height() for c in self.children] + [0])
        self._height_cache = height

        # if not self._is_recursive:
        #     self._is_recursive = False
        return self._height_cache

    def height(self):
        return self._height()



def process_diagram(diagram_content):
    nodes = dict()
    edgelist = []
    for cell in diagram_content:
        if cell.get("source") and cell.get("target"):
            edgelist.append(Edge(cell))
        elif cell.get("id") and cell.get("value"):
            node = Node(cell)
            nodes[node.id] = node

    new_edgelist = []
    for edge in edgelist:
        source_node = nodes[edge.source]
        target_node = nodes[edge.target]
        if source_node.is_sys():
            if "sys" in source_node.var.lower():
                latent_usr_node_var = source_node.var.lower().replace("sys",
                                                               "usr").upper()
            elif "system" in source_node.var.lower():
                latent_usr_node_var = source_node.var.lower().replace("system",
                                                               "user").upper()
            else:
                latent_usr_node_var = "USR_LATENT_" + source_node.var

            if "usr" in target_node.var.lower():
                latent_sys_node_var = target_node.var.lower().replace("usr",
                                                               "sys").upper()
            elif "user" in target_node.var.lower():
                latent_sys_node_var = target_node.var.lower().replace("user",
                                                               "system").upper()
            else:
                latent_sys_node_var = "SYS_LATENT_" + target_node.var

            if latent_usr_node_var not in nodes:
                latent_usr_node = deepcopy(source_node)
                latent_usr_node.var = latent_usr_node.head =\
                    latent_usr_node.text = latent_usr_node.id =\
                    latent_usr_node_var
                nodes[latent_usr_node_var] = latent_usr_node
            else:
                latent_usr_node = nodes[latent_usr_node_var]

            if latent_sys_node_var not in nodes:
                latent_sys_node = deepcopy(target_node)
                latent_sys_node.var = latent_sys_node.head =\
                    latent_sys_node.text = latent_sys_node.id =\
                    latent_sys_node_var
                nodes[latent_sys_node_var] = latent_sys_node
            else:
                latent_sys_node = nodes[latent_sys_node_var]

            new_edgelist.append(Edge({"source": source_node.var,
                                      "target": latent_usr_node.var}))
            new_edgelist.append(Edge({"source": latent_usr_node.var,
                                      "target": latent_sys_node.var}))
            new_edgelist.append(Edge({"source": latent_sys_node.var,
                                      "target": target_node.var}))

            latent_sys_node.dfs_step = latent_usr_node.dfs_step =\
                target_node.dfs_step = source_node.dfs_step
        else:
            new_edgelist.append(Edge({"source": source_node.var,
                                      "target": target_node.var}))

    new_nodes = {}
    for node_id, node in nodes.items():
        if node.var not in new_nodes:
            new_nodes[node.var] = node

    for edge in new_edgelist:
        source_node = new_nodes[edge.source]
        target_node = new_nodes[edge.target]
        source_node.children.append(target_node)
    for node in new_nodes.values():
        node.id = node.var
    nodes = new_nodes

    if not nodes:
        return {}

    root_node = max(nodes, key=lambda k: nodes[k].height())
    root_node = nodes[root_node]

    dfs_visited = []
    def reorder_dfs(node:Node, dfs_ix):
        if node.id in dfs_visited:
            return dfs_ix - 1
        dfs_visited.append(node.id)
        node.dfs_step = f"{dfs_ix}_{node.dfs_step}"
        for child in node.children:
            dfs_ix = reorder_dfs(child, dfs_ix+1)
        return dfs_ix

    reorder_dfs(root_node, 0)

    return nodes


def get_diagrams_info():
    tree = etree.parse(XML_FNAME)

    diag2graph = dict()
    import base64, zlib
    from urllib.parse import unquote
    for diagram_el in tree.getroot().getchildren():
        diagram_name = diagram_el.get("name")
        diagram_el = etree.fromstring(unquote(
            zlib.decompress(base64.b64decode(diagram_el.text), -15).decode(
                "utf-8")))
        diagram_content = diagram_el.getchildren()[0].getchildren()

        diag_nodes = process_diagram(diagram_content)

        diag2graph[diagram_name] = diag_nodes
    return diag2graph


def proc_diag(diag_name, curr_diag: Dict[str, Node]):
    states_code = generate_states_enum_code(curr_diag)
    cond_methods_code = generate_cond_methods_code(curr_diag)
    nodes_code = generate_nodes_code(curr_diag)

    boilerplate_text = open(BOILERPLATE_CODE_FNAME).read()
    filled_text = boilerplate_text.replace("{STATES ENUM}", states_code)\
                                  .replace("{METHODS}", cond_methods_code)\
                                  .replace("{TRANSITIONS}", nodes_code)

    diag_fname = to_upper_snake_case(diag_name).lower() + ".py"
    with open(diag_fname, 'w') as f:
        print(filled_text, file=f)

    return diag_fname


def generate_nodes_code(curr_diag: Dict[str, Node]):
    nodes_codes = []
    prev_node_dfs_ix = None
    for node in sorted(curr_diag.values(), key=lambda v: v.dfs_step):
        curr_node_dfs_ix = node.dfs_step.split('_', maxsplit=1)[-1]
        if curr_node_dfs_ix != prev_node_dfs_ix :
            nodes_codes.append('#'+'-'*79 + f"\n# {node.var}")
            prev_node_dfs_ix = curr_node_dfs_ix
        if not node.children:
            continue

        transition_method = node._transition_method()
        if len(node.children) > 1:
            children_args = []
            for tgt_node in node.children:
                tgt_node_state = f"State.{tgt_node.var}"
                children_args.append(f"{tgt_node_state}: {tgt_node.method}")

            args = [f"State.{node.var}",
                    "{" + ',\n     '.join(children_args) + "}"]
        else:
            tgt_node = node.children[0]
            args = [f"State.{node.var}", f"State.{tgt_node.var}",
                    tgt_node.method]

        # node_code = "#" * 80 + f"\n# {node.head}\n"
        node_code = f"simplified_dialogflow.{transition_method}(\n"
        node_code += ',\n'.join([' ' * 4 + a for a in args])
        node_code += ")\n"
        node_code += f"\nsimplified_dialogflow.set_error_successor(" \
                     f"State.{node.var}, State.SYS_ERR)"
        nodes_codes.append(node_code)
    nodes_code = '\n\n'.join(nodes_codes)
    return nodes_code


def generate_request_method_body(method_name):
    body = '\n'.join(["    flag = False",
                      "    raise NotImplementedException()  # YOUR CODE HERE",
                      '    logger.info(f"weekend_request={flag}")',
                      "    return flag"])
    return body


def generate_response_method_body(method_name):
    body = f"""    logger.info("exec {method_name}")
    try:
        state_utils.set_confidence(vars, MUST_CONTINUE_CONFIDENCE)
        state_utils.set_can_continue(vars)
        response_text = ''  # YOUR CODE HERE
        raise NotImplementedException()  # YOUR CODE HERE
        return response_text
    except NotImplementedException:
        raise NotImplementedException('you should implement {method_name}')
    except Exception as exc:
        logger.exception(exc)
        sentry_sdk.capture_exception(exc)
        state_utils.set_confidence(vars, CANNOT_CONTINUE_CONFIDENCE)
        return error_response(vars)"""
    return body



def generate_method_code(method_name):
    code = f"def {method_name}(vars):\n"
    if method_name.endswith("request"):
        code += generate_request_method_body(method_name)
    elif method_name.endswith("response"):
        code += generate_response_method_body(method_name)
    else:
        code += "    raise NotImplementedException()"
    return code


def generate_cond_methods_code(curr_diag):
    cond_methods_codes = []
    for node in curr_diag.values():
        method_code = generate_method_code(node.method)
        cond_methods_codes.append(method_code)
    cond_methods_codes = sorted(set(cond_methods_codes))
    cond_methods_code = '\n\n'.join(cond_methods_codes)
    return cond_methods_code


def generate_states_enum_code(curr_diag):
    states_code = "class State(Enum):"
    states_code_lines = []
    for node_ix, node in curr_diag.items():
        states_code_lines.append(f"{node.var} = auto()")
    states_code_lines.extend(["SYS_ERR = auto()", "USR_ERR = auto()",
                        "SYS_END = auto()", "USR_END = auto()"])
    states_code_lines = sorted(set(states_code_lines))
    states_code += '\n    ' + '\n    '.join(states_code_lines)
    return states_code


def to_upper_snake_case(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    res_text = ''
    for c in text:
        res_text += c if c.isalpha() else '_'
    res_text = res_text.strip('_')
    res_text = res_text.upper()
    res_text = re.sub('_+', '_', res_text)
    return res_text


from flask import request, send_file

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "GET":
        download_diagrams()
        diagrams_info = get_diagrams_info()
        d_names = diagrams_info.keys()

        return render_template('template.html', d_names=d_names)
    if request.method == "POST":
        dname = request.form["d_names"]
        diagrams_info = get_diagrams_info()

        fname = proc_diag(dname, diagrams_info[dname])
        return send_file(fname, as_attachment=True)


app.run(host='0.0.0.0', port=8080)
