import read, copy
from util import *
from logical_classes import *

verbose = 0

class KnowledgeBase(object):
    def __init__(self, facts=[], rules=[]):
        self.facts = facts
        self.rules = rules
        self.ie = InferenceEngine()

    def __repr__(self):
        return 'KnowledgeBase({!r}, {!r})'.format(self.facts, self.rules)

    def __str__(self):
        string = "Knowledge Base: \n"
        string += "\n".join((str(fact) for fact in self.facts)) + "\n"
        string += "\n".join((str(rule) for rule in self.rules))
        return string

    def _get_fact(self, fact):
        """INTERNAL USE ONLY
        Get the fact in the KB that is the same as the fact argument

        Args:
            fact (Fact): Fact we're searching for

        Returns:
            Fact: matching fact
        """
        for kbfact in self.facts:
            if fact == kbfact:
                return kbfact

    def _get_rule(self, rule):
        """INTERNAL USE ONLY
        Get the rule in the KB that is the same as the rule argument

        Args:
            rule (Rule): Rule we're searching for

        Returns:
            Rule: matching rule
        """
        for kbrule in self.rules:
            if rule == kbrule:
                return kbrule

    def kb_add(self, fact_rule):
        """Add a fact or rule to the KB
        Args:
            fact_rule (Fact or Rule) - Fact or Rule to be added
        Returns:
            None
        """
        # printv("Adding {!r}", 1, verbose, [fact_rule])
        if isinstance(fact_rule, Fact):
            if fact_rule.asserted == False or fact_rule not in self.facts:
                # print("if fact_rule not in self.facts:")
                self.facts.append(fact_rule)
                for rule in self.rules:
                    self.ie.fc_infer(fact_rule, rule, self)
                    
            else:
                # print("if fact_rule in self.facts:")
                if fact_rule.supported_by:
                    ind = self.facts.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.facts[ind].supported_by.append(f)
                else:
                    ind = self.facts.index(fact_rule)
                    self.facts[ind].asserted = True
        elif isinstance(fact_rule, Rule):
            if fact_rule.asserted == False or fact_rule not in self.rules:
                self.rules.append(fact_rule)
                for fact in self.facts:
                    self.ie.fc_infer(fact, fact_rule, self)
                    
            else:
                if fact_rule.supported_by:
                    ind = self.rules.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.rules[ind].supported_by.append(f)
                else:
                    ind = self.rules.index(fact_rule)
                    self.rules[ind].asserted = True

    def kb_assert(self, fact_rule):
        """Assert a fact or rule into the KB

        Args:
            fact_rule (Fact or Rule): Fact or Rule we're asserting
        """
        # printv("Asserting {!r}", 0, verbose, [fact_rule])
        self.kb_add(fact_rule)

    def kb_ask(self, fact):
        """Ask if a fact is in the KB

        Args:
            fact (Fact) - Statement to be asked (will be converted into a Fact)

        Returns:
            listof Bindings|False - list of Bindings if result found, False otherwise
        """
        # print("Asking {!r}".format(fact))
        if factq(fact):
            f = Fact(fact.statement)
            bindings_lst = ListOfBindings()
            # ask matched facts
            for fact in self.facts:
                binding = match(f.statement, fact.statement)
                if binding:
                    bindings_lst.add_bindings(binding, [fact])

            return bindings_lst if bindings_lst.list_of_bindings else []

        else:
            # print("Invalid ask:", fact.statement)
            return []
        

    def kb_reasoning_behind_fact(self, fact):
        """Outputs the reasoning chain behind a fact.

        Args:
            fact (Fact) - Fact to output the reasoning chain for.

        Returns:
            str - A string showing the reasoning chain behind the fact in question.
        """
        fact_in_kb = self._get_fact(fact)  # Get the actual fact object from the KB.
        if not fact_in_kb:
            return f"Fact {fact.statement} not found in the knowledge base."

        reasoning_chain = []
        self._build_reasoning_chain(fact_in_kb, reasoning_chain)
        return "\n".join(reasoning_chain)

    def _build_reasoning_chain(self, fact, reasoning_chain):
        """Helper function to recursively build the reasoning chain for a fact."""
        for support in fact.supported_by:
            supporting_fact, supporting_rule = support
            reasoning = f"({supporting_fact.statement}) > ({supporting_rule.rhs})"
            reasoning_chain.append(reasoning)
            
            # Recurse into the supporting fact for deeper reasoning.
            self._build_reasoning_chain(supporting_fact, reasoning_chain)
    def kb_retract(self, fact_rule):
        """Retract a fact or a rule from the KB

        Args:
            fact_rule (Fact or Rule) - Fact or Rule to be retracted

        Returns:
            None
        """
        if isinstance(fact_rule, Fact):
            fact = self._get_fact(fact_rule)
            if fact and fact.asserted:  # Only retract asserted facts
                self._retract_fact(fact)

        elif isinstance(fact_rule, Rule):
            rule = self._get_rule(fact_rule)
            if rule and rule.asserted:  # Only retract asserted rules
                self._retract_rule(rule)

    def _retract_fact(self, fact):
        # Ensure the fact is only retracted if it is not supported by other facts/rules
        if not fact.supported_by:
            # Remove the fact from the knowledge base
            if fact in self.facts:
                self.facts.remove(fact)

            # Retract any facts or rules that this fact supports
            for supported_fact in fact.supports_facts:
                supported_fact.supported_by = [
                    pair for pair in supported_fact.supported_by if pair[0] != fact
                ]
                # If the supported fact is no longer supported, retract it
                if not supported_fact.supported_by and not supported_fact.asserted:
                    self._retract_fact(supported_fact)

            for supported_rule in fact.supports_rules:
                supported_rule.supported_by = [
                    pair for pair in supported_rule.supported_by if pair[0] != fact
                ]
                # If the supported rule is no longer supported, retract it
                if not supported_rule.supported_by and not supported_rule.asserted:
                    self._retract_rule(supported_rule)

    def _retract_rule(self, rule):
        # Ensure the rule is only retracted if it is not supported by other facts/rules
        if not rule.supported_by:
            # Remove the rule from the knowledge base
            if rule in self.rules:
                self.rules.remove(rule)

            # Retract any facts or rules that this rule supports
            for supported_fact in rule.supports_facts:
                supported_fact.supported_by = [
                    pair for pair in supported_fact.supported_by if pair[1] != rule
                ]
                # If the supported fact is no longer supported, retract it
                if not supported_fact.supported_by and not supported_fact.asserted:
                    self._retract_fact(supported_fact)

            for supported_rule in rule.supports_rules:
                supported_rule.supported_by = [
                    pair for pair in supported_rule.supported_by if pair[1] != rule
                ]
                # If the supported rule is no longer supported, retract it
                if not supported_rule.supported_by and not supported_rule.asserted:
                    self._retract_rule(supported_rule)


class InferenceEngine(object):
    def fc_infer(self, fact, rule, kb):
        """Forward-chaining to infer new facts and rules

        Args:
            fact (Fact) - A fact from the KnowledgeBase
            rule (Rule) - A rule from the KnowledgeBase
            kb (KnowledgeBase) - A KnowledgeBase

        Returns:
            Nothing
        """
        # print('Attempting to infer from {!r} and {!r} => {!r}'.format(fact.statement, rule.lhs, rule.rhs))
        ####################################################
        # Student code goes here
        # Check if the fact matches the first statement in the rule's LHS
        bindings = match(fact.statement, rule.lhs[0])
        # print("rule: ", rule)
        # print("fact: ", fact)
        # If there's a match, proceed with inference
        if bindings:
            # If there are more statements on the LHS of the rule, we create a new inferred rule
            if len(rule.lhs) > 1:
                # print("***************create a new inferred rule")
                new_lhs = [instantiate(statement, bindings) for statement in rule.lhs[1:]]
                new_rhs = instantiate(rule.rhs, bindings)
                new_rule = Rule([new_lhs, new_rhs], [[fact, rule]])
                # print(new_rule)
                # Add the new rule to the knowledge base and update support relations
                fact.supports_rules.append(new_rule)
                rule.supports_rules.append(new_rule)
                kb.kb_add(new_rule)

            else:  # If the LHS of the rule has only one statement, we create a new fact
                # print("**************create a new inferred fact")
                new_fact = Fact(instantiate(rule.rhs, bindings), [[fact, rule]])
                # print(new_fact)
                # print(kb)
                # Add the new fact to the knowledge base and update support relations
                fact.supports_facts.append(new_fact)
                rule.supports_facts.append(new_fact)
                kb.kb_add(new_fact)
                # print("==========")
                # print(kb)
        # print("fc_infer finished")