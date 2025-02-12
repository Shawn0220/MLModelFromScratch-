def ask(var, value, evidence, bn):
    """
    Calculates the probability of a hypothesis given evidence: P(H | E)
    
    Parameters:
    var (str): The name of the hypothesis variable.
    value (bool): The value of the hypothesis (True or False).
    evidence (dict): A dictionary of evidence variables and their values.
    bn (BayesNet): The BayesNet object representing the network.

    Returns:
    float: The probability P(H | E).
    """
    def joint_probability(variables, known_values):
        # Base case: If no more variables to process, return 1
        if not variables:
            return 1.0
        
        # Get the first variable to process
        current_var = variables[0]
        remaining_vars = variables[1:]
        
        if current_var in known_values:
            # If the current variable has a known value, use its probability and recurse
            node = bn.get_var(current_var)
            prob = node.probability(known_values[current_var], known_values)
            return prob * joint_probability(remaining_vars, known_values)
        else:
            # If the current variable's value is unknown, sum over both possibilities
            node = bn.get_var(current_var)
            
            # Case where current_var is True
            known_values[current_var] = True
            prob_true = node.probability(True, known_values) * joint_probability(remaining_vars, known_values)
            
            # Case where current_var is False
            known_values[current_var] = False
            prob_false = node.probability(False, known_values) * joint_probability(remaining_vars, known_values)
            
            # Clean up the temporary evidence for this variable
            del known_values[current_var]
            
            return prob_true + prob_false
    
    # Get the list of all variables in the Bayes Net
    variables = bn.variable_names
    
    # Calculate P(H, E)
    evidence[var] = value
    numerator = joint_probability(variables, evidence)
    del evidence[var]  # Clean up the added hypothesis from evidence
    
    # Calculate P(E)
    denominator = joint_probability(variables, evidence)
    
    # Return P(H | E) = P(H, E) / P(E)
    return numerator / denominator