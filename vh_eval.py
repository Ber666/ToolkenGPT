import json
import sys
sys.path.append('virtualhome/src/virtualhome/simulation')
import virtualhome
import evolving_graph.utils as utils
from evolving_graph.scripts import *
from evolving_graph.execution import *
from evolving_graph.environment import *
from evolving_graph.preparation import *

def verify_script(executor, my_script, init_state, final_state, verbose=False):

    def eq(a, b):
        if isinstance(a, list) and isinstance(b, list):
            return set(a).issubset(set(b)) and set(b).issubset(set(a))
        else:
            return a == b

    return_values = {"state": True, "desc": ""}
    difference_list = []
    final_state_dict = final_state.to_dict()
    init_state_dict = init_state.to_dict()
    init_state_dict["nodes"] = sorted(init_state_dict["nodes"], key=lambda x: x["id"])
    final_state_dict["nodes"] = sorted(final_state_dict["nodes"], key=lambda x: x["id"])
    for node_init, node_final in zip(final_state_dict["nodes"], init_state_dict["nodes"]):
        if any([node_init[key] != node_final[key] for key in node_final.keys()]):
            difference_list.append((node_final, node_init))
    if len(difference_list) == 0:
        return_values = {"state": False, "desc": "difference list too short"}
        return return_values
    if len(difference_list) >= 3:
        return_values = {"state": False, "desc": "The difference list is too long or too short"}
        return return_values
    return_values = {"state": True, "desc": ""}
    cur_state = init_state
    for ind in range(len(my_script)):
        if verbose:
            print(ind, my_script[ind])
        # print(my_script[ind])
        # status = executor.check_one_step(my_script.from_index(ind), cur_state)
        # print(status)
        # try:
        try:
            status, cur_state = executor.execute_one_step(my_script.from_index(ind), cur_state, in_place=False)
        except Exception as e:
            return_values["state"] = False
            return_values["desc"] = f"Execution internal failed: {my_script[ind]} | {e}"
            return return_values
            # chars = cur_state.get_nodes_by_attr('class_name', 'character')
        # except Exception as e:
        #     print("a bug in virtualhome code" + e)
        #     status, cur_state = False, cur_state
        if status is False:
            info = executor.info.get_error_string()
            return_values["state"] = False
            return_values["desc"] = f"Execution failed: {my_script[ind]} | {info}"
            return return_values
    final_script_nodes = cur_state.to_dict()["nodes"]
    for node_init, node_final in difference_list:
        id = node_final["id"]
        for final_script_node in final_script_nodes:
            if final_script_node["id"] == id:
                break
        if any([not eq(node_final[key], final_script_node[key]) for key in node_final.keys()]):
            return_values["state"] = False
            return_values["desc"] = "The final state is not correct: " + str(node_final) + " " + str(final_script_node)
            return return_values
    else:
        return return_values
    

def get_desc(init_state=None, graph_file_name=None, goal=None, goal_fine=None, script_file_name=None, obj_list=[]):

    assert script_file_name is not None or (goal is not None and goal_fine is not None)
    if script_file_name is not None:
                
        with open(script_file_name, 'r') as f:
            script = f.readlines()
        goal = script[0].strip()
        goal_fine = script[1].strip()
        # print(script, goal, goal_fine)

    assert init_state is not None or graph_file_name is not None
    if graph_file_name is not None:
        graph = json.load(open(graph_file_name))
        name_equivalence = utils.load_name_equivalence()
        graph_init = EnvironmentGraph(graph["init_graph"])
        graph_final = EnvironmentGraph(graph["final_graph"])
        init_state = EnvironmentState(graph_init, name_equivalence)
        final_state = EnvironmentState(graph_final, name_equivalence)

    Desc = ""
    chars = init_state.get_nodes_by_attr('class_name', 'character')
    if len(chars) > 0:
        char = chars[0]

    inside = init_state.get_nodes_from(char, Relation.INSIDE)
    if len(inside) > 0:
        Desc += "I am in " + str([n.class_name for n in inside]) + ". "
    holding = init_state.get_nodes_from(char, Relation.HOLDS_RH) + init_state.get_nodes_from(char, Relation.HOLDS_LH)
    if len(holding) > 0:
        Desc += "I am holding " + str([n.class_name for n in holding]) + ". "
    on = init_state.get_nodes_from(char, Relation.ON)
    if len(on) > 0:
        Desc += "I am on " + str([n.class_name for n in on]) + ". "
    # print(Desc)

    final_state_dict = final_state.to_dict()
    init_state_dict = init_state.to_dict()
    assert len(init_state_dict['nodes']) == len(final_state_dict['nodes'])
    # %%
    init_state.get_nodes_by_attr('class_name', 'character')
    env_list = set(n["class_name"] for n in init_state_dict["nodes"] if n['category'] not in ['Walls','Floor'])
    if len(obj_list) > 0: # need to filter
        env_list = env_list.intersection(set(obj_list))
    print("env list: ", len(env_list))
    Desc += "The objects I can manipulate are " + str(list(env_list)) + ".\n"
    Desc += "Goal:\n" + goal.strip() + "\nHint:\n" + goal_fine.strip() + "\nPlan:\n"

    return Desc

if __name__ == "__main__":
    # graph_path = 'virtualhome/src/virtualhome/dataset/programs_processed_precond_nograb_morepreconds/init_and_final_graphs/TrimmedTestScene1_graph/results_intentions_march-13-18/file3_1.json'
    # script_path = 'virtualhome/src/virtualhome/dataset/programs_processed_precond_nograb_morepreconds/executable_programs/TrimmedTestScene1_graph/results_intentions_march-13-18/file3_1.txt'
    graph_path = "virtualhome/src/virtualhome/dataset/programs_processed_precond_nograb_morepreconds/init_and_final_graphs/TrimmedTestScene4_graph/results_intentions_march-13-18/file71_1.json"
    script_path = "virtualhome/src/virtualhome/dataset/programs_processed_precond_nograb_morepreconds/executable_programs/TrimmedTestScene4_graph/results_intentions_march-13-18/file71_1.txt"
    graph = json.load(open(graph_path))

    name_equivalence = utils.load_name_equivalence()

    graph_init = EnvironmentGraph(graph["init_graph"])
    graph_final = EnvironmentGraph(graph["final_graph"])
    init_state = EnvironmentState(graph_init, name_equivalence)
    final_state = EnvironmentState(graph_final, name_equivalence)


    with open(script_path, 'r') as f:
        script = f.readlines()
    goal = script[0].strip()
    goal_fine = script[1].strip()
    # print(script, goal, goal_fine)

    script = read_script(script_path)
    # print(script[0])
    vh_script = Script(script)

    executor = ScriptExecutor(graph_init, name_equivalence)

    chars = init_state.get_nodes_by_attr('class_name', 'character')
    if len(chars) > 0:
        char = chars[0]

    # print(verify_script(executor, vh_script, init_state, final_state))
    # print(get_desc(init_state, goal, goal_fine))
    # %%
