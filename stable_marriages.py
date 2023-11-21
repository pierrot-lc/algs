"""Implement the matching algorithm for the "Stable marriage problem".
The matching is done using the "Gale-Shapley algorithm".

The implementation is done using pure numpy operations.
The verification of a stable marriage is provided with pure numpy operations.

See https://en.wikipedia.org/wiki/Stable_marriage_problem.
See https://en.wikipedia.org/wiki/Gale%E2%80%93Shapley_algorithm.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Group:
    """A group of people with preferences over the other group.

    Attributes
    ----------
    size_self : The size of the group.
    size_others : The size of the other group.
    preferences : The preferences of the group over the other group.
        Shape of [size_self, size_others].
    matched : Currently matched people, boolean array.
        Shape of [size_self,].
    current_matches : Current matches, int array of ids.
        Shape of [size_self,].
    """

    size_self: int
    size_others: int
    preferences: np.ndarray
    matched: np.ndarray
    current_matches: np.ndarray

    @classmethod
    def random_init(
        cls, size_self: int, size_others: int, seed: int | None = None
    ) -> "Group":
        rng = np.random.default_rng(seed)
        preferences = np.array([rng.permutation(size_others) for _ in range(size_self)])
        return Group(
            size_self=size_self,
            size_others=size_others,
            preferences=preferences,
            matched=np.zeros(size_self, dtype=bool),
            current_matches=np.zeros(size_self, dtype=int) - 1,
        )


def do_one_round(group_1: Group, group_2: Group):
    """Do one round of the matching algorithm.
    Group 1 proposes to group 2, and group 2 accepts or rejects.

    The round is simulated by following those steps:
        1. Update the current cursor of G1 for persons currently not matched.
        2. Extract ids of G2 persons based on the cursor of each persons in G1.
           Note that we also take the current proposal of persons already matched.
        3. Build the table of proposals of shape [G2_size, G1_size] where
           "proposals[i, j] == 1" <=> "person j in G1 is proposing by person i in G2".
        4. Reorder each proposals to follow the preferences of each person in G2.
        5. Add a fictive column to the proposals to take into account persons in G2
           whose current best proposal is nobody.
        6. Select the best candidate for each person in G2 by taking the argmax over
           their proposals. This becomes their new best proposal.
        7. Each candidate in G1 and G2 gets its `matched` value updated.
    """
    group_1_size = group_1.size_self
    group_2_size = group_2.size_self
    unmatched_g1 = ~group_1.matched
    group_1.current_matches[unmatched_g1] += 1

    # Id of persons in group 2 each person in group 1 will propose to.
    candidates_g2 = np.take_along_axis(
        group_1.preferences,
        group_1.current_matches[:, None],
        axis=1,
    ).flatten()  # Shape of [group_1_size,].
    ids_g1 = np.arange(group_1.size_self)

    # `proposals[i, j] = 1` <=> "person i from G2 is proposed from person j from G1"
    proposals = np.zeros((group_2.size_self, group_1.size_self), dtype=int)
    proposals[candidates_g2, ids_g1] = 1

    # Permute the proposals for each person in G2 by their personal preferences.
    proposals = np.take_along_axis(proposals, group_2.preferences, axis=1)
    # Add the column representing "no match" for G2.
    proposals = np.concatenate(
        (proposals, np.zeros((group_2_size, 1), dtype=int)), axis=1
    )
    # Add the current matches of G2.
    ids_g2 = np.arange(group_2_size)
    proposals[ids_g2, group_2.current_matches] = 1
    # The choosen candidate is the first one to be equal to 1.
    group_2.current_matches = np.argmax(proposals, axis=1)

    # Fetch the id of each selected candidates.
    group_1_candidates = np.concatenate(
        (group_2.preferences, np.zeros((group_2_size, 1), dtype=int)),
        axis=1,
    )
    group_1_candidates[:, -1] = group_1_size
    selected_candidates_1 = np.take_along_axis(
        group_1_candidates, group_2.current_matches[:, None], axis=1
    ).flatten()

    # Update infos in 1, don't forget that some selected candidates may be the "nobody" candidate.
    group_1.matched = np.zeros(group_1_size + 1, dtype=bool)
    group_1.matched[selected_candidates_1] = True
    group_1.matched = group_1.matched[:-1]  # Remove fictive "nobody" column.

    # Update infos in 2, a person in G2 has a match if its selected candidate is not the "nobody" candidate.
    group_2.matched = selected_candidates_1 != group_1_size


def solve_gale_shapley(size_group_1: int, size_group_2: int):
    group_1 = Group.random_init(size_group_1, size_group_2)
    group_2 = Group.random_init(size_group_2, size_group_1)

    # Initially the group 2 is not matched with anybody.
    group_2.current_matches = size_group_1

    while not np.all(group_1.matched):
        do_one_round(group_1, group_2)

    return group_1, group_2


def is_group_one_stable(group_1: Group, group_2: Group) -> bool:
    """Whether the marriage is stable when seen from group_1 over group_2.

    The algorithm is quite complex. Here are the steps:
        1. Id of G2 persons matched for each person in G1 (M).
        2. Preferences of each person M (N).
        3. For each preferred person N of M, compute the rank of the person M in
           the preferences of N.
        4. For each preferred person N of M, compute the rank of the matched person
           G2 of N in the preferences of N.
        5. For each person N in the preferences of M, compute the difference of the
           ranks between its current match and the rank of the person M.
        6. Make sure that for every person M, the difference is negative for all
           persons N placed above the current match of M.
    """
    # 1. Id of G2 persons matched for each person in G1.
    # Shape of [group_1_size,], values range between [0, group_2_size - 1].
    # Let call it "M".
    matches_of_G1 = group_1.preferences[
        np.arange(group_1.size_self), group_1.current_matches
    ]

    # 2. Preferences of each person M.
    # Shape of [group_1_size, group_1_size], values are ids of persons in G1.
    preferences_of_M = group_2.preferences[matches_of_G1]
    # "preferences_of_matches_of_G1[i] = V"
    #   <=>
    # "The ith person in G1 is matched with someone in G2 having this vector V of preferences."

    # 3. For each preferred person N of M, compute the rank of the person M in
    # the preferences of N.
    # Shape of [group_1_size, group_1_size], values are ids of persons in G2.
    ordered_preferences_of_G1 = np.argsort(group_1.preferences, axis=1)
    rank_of_M_in_N = ordered_preferences_of_G1[preferences_of_M][
        np.arange(group_1.size_self), :, matches_of_G1
    ]
    # "rank_of_M_in_N[i, j] = r"
    #   <=>
    # "The ith person in G1 is matched with someone k in G2 whose jth preference
    # is someone in G1 having k ranked r in its preferences."

    # 4. For each preferred person N of M, compute the rank of the matched person G2 of N
    # in the preferences of N.
    # Shape of [gorup_1_size, group_1_size], values are ids of persons in G2.
    rank_of_N_current_match_in_N = group_1.current_matches[preferences_of_M]
    # "rank_of_N_current_match_in_N[i, j] = r"
    #   <=>
    # "The ith person in G1 is matched with someone in G2 whose jth preference
    # is someone in G1 being matched with someone (else) in G2 ranked r in its preferences."

    # For the marriage to be stable, the persons N that are above in the preferences
    # of M w.r.t. its current match should be matched with someone above in their
    # preferences than M.
    # 5. For each person N in the preferences of M, compute the difference of the
    # ranks between its current match and the rank of the person M.
    diff_ranks = rank_of_N_current_match_in_N - rank_of_M_in_N
    # "diff_ranks[i, j] > 0"
    #   =>
    # "The ith person in M has a jth person in N whose current match is ranked
    # worse than this ith person the the jth person preferences."

    # 6. Make sure that for every person M, the difference is negative for all
    # persons N placed above the current match of M.
    # Note: for ranks, here, lower is better.
    # Note: the current match of M have a difference of 0.
    positive_strict = np.cumsum(diff_ranks > 0, axis=1)
    positive = np.cumsum(diff_ranks >= 0, axis=1)
    return np.all(positive - positive_strict >= 0)


if __name__ == "__main__":
    for _ in range(100):
        group_1, group_2 = solve_gale_shapley(50, 50)
        assert np.all(group_1.matched), "Not all persons in G1 have a match."
        assert np.all(group_2.matched), "Not all persons in G2 have a match."
        assert is_group_one_stable(group_1, group_2), "Matches of G1 are not stable."
        assert is_group_one_stable(group_2, group_1), "Matches of G2 are not stable."
