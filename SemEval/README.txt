The enclosed folders and files comprise all original data from:

    Peterson, J.*, Dawn, C.*, & Griffiths, T. (2020). Parallelograms revisited: Exploring the 
    limitations of vector space models for simple analogies. Cognition. (*equal contribution),

where detailed methods describing how the data was collected can be found.

For additional questions, email peterson.c.joshua@gmail.com
http://www.joshpeterson.io/
https://github.com/jcpeterson


The contents of this data archive are:


   FOLDER     : experiment1_data_completions

   DESCRIPTION: Total completion counts from all study participants for experiments 1a, 1b,
                and 1c (files: experiment1a_counts.csv, experiment1b_counts.csv, and 
                experiment1c_counts.csv). In each file, the first column shows the three
                word prompt shown to participants (e.g., ('father', 'son', 'mother')),
                corresponding to the question "father is to son as mother is to ?". The next
                three columns prompt_w1, prompt_w2, and prompt_w3 provide prompt words 
                individually (e.g., 'father', 'son', and 'mother'). The remaining columns
                include all completion words given by partipants across all prompts, which
                range from over 600 to over 3000 words across files. For each of these
                columns, each row gives the total completion counts for each prompt. The file
                is highly sparse, and suitable only for computational analysis. For easier
                browsing of the count data, see "readable_counts.csv", which shows results
                in short form, such as:

                    Prompt: father : son :: mother : ?
                    Responses: daughter (30), father (1)


   FOLDER     : experiment2_data_relational_similarity

   DESCRIPTION: Raw (per subject, per trial; "raw_relsim_ratings.csv") and average (per
                stimulus; "mean_relsim_ratings.csv") relational similarity ratings. The
                raw ratings file contains the columns:

                    sub_id          : unique ID of the subject
                    trial_idx       : ordered index of the trial
                    pair_left_word1 : first word from word pair show on the left
                    pair_left_word2 : second word from word pair show on the left
                    pair_left_type  : semantic classification ID of the left pair
                    pair_right_word1: first word from word pair show on the right
                    pair_right_word2: second word from word pair show on the right
                    pair_right_type : semantic classification ID of the right pair
                    rt              : time taken in milliseconds to complete trial
                    time_elapsed    : total time (ms) since experiment started 
                    sim_rating      : relational similarity rating between 1 and 7
                    attention_check : whether the trial was an attention check
                    comparison_type : whether the two word pairs have the same
                                      semantic subtype (within-subtype), different 
                                      subtypes (between-subtype), or different types
                                      (between-type)

                The average ratings file contains the columns:

                    relation1      : semantic classification ID of word pair 1
                    relation2      : semantic classification ID of word pair 2
                    comparison_type: (same as above)
                    pair1_word1    : first word from word pair 1
                    pair1_word2    : second word from word pair 1
                    pair2_word1    : first word from word pair 2
                    pair2_word2    : second word from word pair 2
                    mean_rating    : average relational similarity rating
                    num_ratings    : total number of ratings for each stimulus

                (Semantic classification ID mappings are given in the appendix below.)


   FOLDER     : experiment3_data_symmetry

   DESCRIPTION: Raw (per subject, per trial; "raw_asymmetry_ratings.csv") and average
                (per stimulus; "mean_asymmetry_ratings.csv") relational similarity ratings.
                The difference between these ratings and those above is that word pairs are
                presented sequentially in either forward or backward order. The raw ratings
                file contains the columns:

                    sub_num    : unique ID of the subject
                    trial_num  : ordered index of the trial
                    trial_type : same as "comparison type" above
                    relation1  : semantic classification ID of word pair 1
                    relation2  : semantic classification ID of word pair 2
                    pair1_word1: first word from word pair 1
                    pair1_word2: second word from word pair 1
                    pair2_word1: first word from word pair 2
                    pair2_word2: second word from word pair 2
                    rating     : relational similarity rating between 1 and 7
                    RT         : time taken in milliseconds to complete trial

                The average ratings file contains the columns:

                    relation1      : semantic classification ID of word pair 1
                    relation2      : semantic classification ID of word pair 2
                    comparison_type: (same as above)
                    pair1_word1    : first word from word pair 1
                    pair1_word2    : second word from word pair 1
                    pair2_word1    : first word from word pair 2
                    pair2_word2    : second word from word pair 2
                    forward_rating : rating given order: pair1, pair2
                    forward_n      : total number of forward ratings
                    backward_rating: rating given order: pair2, pair1
                    backward_n     : total number of backward ratings

                (Semantic classification ID mappings are given in the appendix below.)


   FOLDER     : experiment4_data_triads

   DESCRIPTION: Raw (per subject, per trial; "raw_triad_ratings.csv") and average (per 
                stimulus; "mean_traid_ratings.csv") analogy quality ratings. The raw ratings
                file contains the columns:

                    sub_num   : unique ID of the subject
                    trial_num : ordered index of the trial
                    trial_type: "real" or attention check trial
                    wordA     : A word of the analogy AB:CD
                    wordB     : B word of the analogy AB:CD
                    wordC     : C word of the analogy AB:CD
                    wordD     : D word of the analogy AB:CD
                    rating    : quality rating for the analogy
                    RT        : time taken in milliseconds to complete trial

                The average ratings file contains the columns:

                    wordA      : A word of the analogy AB:CD
                    wordB      : B word of the analogy AB:CD
                    wordC      : C word of the analogy AB:CD
                    wordD      : D word of the analogy AB:CD
                    mean_rating: average quality rating for the analogy
                    num_ratings: total number of ratings for each stimulus


APPENDIX: Semantic class IDs and descriptions

    Reproduced from the SemiEval 2020 Dataset:

        https://sites.google.com/site/semeval2012task2/download
        https://dl.acm.org/doi/pdf/10.5555/2387636.2387693

1, a, CLASS-INCLUSION, Taxonomic, Y is a kind/type/instance of X
1, b, CLASS-INCLUSION, Functional, Y functions as an X
1, c, CLASS-INCLUSION, Singular Collective, a Y is one item in a collection/group of X
1, d, CLASS-INCLUSION, Plural Collective, Y are items in a collection/group of X
1, e, CLASS-INCLUSION, ClassIndividual, Y is a specific X
2, a, PART-WHOLE, Object:Component, a Y is a part of an X
2, b, PART-WHOLE, Collection:Member, X is made from a collection of Y
2, c, PART-WHOLE, Mass:Potion, X may be divided into Y
2, d, PART-WHOLE, Event:Feature, Y is typically found at an event such as X
2, e, PART-WHOLE, Activity:Stage, X is one step/action/part of the actions in Y
2, f, PART-WHOLE, Item:Topological Part, Y is one of the areas/locations of X
2, g, PART-WHOLE, Object:Stuff, X is made of / is comprised of Y
2, h, PART-WHOLE, Creature:Possession, X possesses/owns/has Y
2, i, PART-WHOLE, Item:Distinctive Nonpart, X is devoid of / cannot have Y
2, j, PART-WHOLE, Item:Ex-part/Ex-possession, an X once had/owned/possessed Y but no longer
3, a, SIMILAR, Synonymity, an X and Y are a similar type of action/thing/attribute
3, b, SIMILAR, Dimensional Similarity, an X and Y are two kinds in a category of actions/things/attributes
3, c, SIMILAR, Dimensional Excessive, Y is an excessive form of X
3, d, SIMILAR, Dimensional Naughty, Y is an unacceptable form of X
3, e, SIMILAR, Conversion, X will become / be converted into Y
3, f, SIMILAR, Attribute Similarity, X and Y both have a similar attribute or feature
3, g, SIMILAR, Coordinates, X and Y are two distinct objects in the same category
3, h, SIMILAR, Change, an X is an increase/decease in Y
4, a, CONTRAST, Contradictory, Something cannot be/have/do X and Y at the same time
4, b, CONTRAST, Contrary, X and Y are contrary / opposite to each other
4, c, CONTRAST, Reverse, X is the reverse act of Y / X may be undone by Y
4, d, CONTRAST, Directional, X is the opposite direction from Y
4, e, CONTRAST, Incompatible, Being X is incompatible with being Y
4, f, CONTRAST, Asymmetric Contrary, X and Y are at opposite ends of the same scale but X is more extreme than Y
4, g, CONTRAST, Pseudoantonym, X is similar to the opposite of Y but X is not truly the opposite of Y
4, h, CONTRAST, Defective, an X is is a defect in Y
5, a, ATTRIBUTE, ItemAttribute(noun:adjective), an X has the attribute Y
5, b, ATTRIBUTE, Object Attribute:Condition, something that is X may be Y
5, c, ATTRIBUTE, ObjectState(noun:noun), an X exists in the state of Y
5, d, ATTRIBUTE, Agent Attribute:State, a person who is X often is in a state of Y
5, e, ATTRIBUTE, Object:Typical Action (noun.verb), an X will typically Y
5, f, ATTRIBUTE, Agent/ObjectAttribute:Typical Action, something/someone that is X will typically Y
5, g, ATTRIBUTE, Action:Action Attribute, X is a Y kind of action
5, h, ATTRIBUTE, Action:Object Attribute, someone will X an object that is Y
5, i, ATTRIBUTE, Action:Resultant Attribute (verb:noun/adjective), the action X results Y or things that are Y
6, a, NON-ATTRIBUTE, Item:Nonattribute (noun:adjective), an X cannot have attribute Y; Y is antithetical to being X
6, b, NON-ATTRIBUTE, ObjectAttribute:Noncondition (adjective:adjective), something that is X cannot be Y
6, c, NON-ATTRIBUTE, Object:Nonstate (noun:noun), Y describes a condition or state that is usually absent from X
6, d, NON-ATTRIBUTE, Attribute:Nonstate (adjective:noun), someone/something who is X cannot be Y or be in the state of Y
6, e, NON-ATTRIBUTE, Objects:Atypical Action (noun:verb), an X is unlikely to Y
6, f, NON-ATTRIBUTE, Agent/Object Attribute: Atypical Action (adjective:verb), someone/something who is X is unlikely to Y
6, g, NON-ATTRIBUTE, Action:Action Nonattribute, X cannot be done in a Y manner
6, h, NON-ATTRIBUTE, Action:Object Nonattribute, the result of action X does not produce an object with Y
7, a, CASE RELATIONS, Agent:Object, an X makes Y / an X uses Y to make an item
7, b, CASE RELATIONS, Agent:Recipient, a Y receives an item/knowledge/service from X
7, c, CASE RELATIONS, Agent:Instrument, an X uses Y to perform their role
7, d, CASE RELATIONS, Action:Object, someone perform the action X on Y
7, e, CASE RELATIONS, Action:Recipient, to X is to have a Y receive some object/service/idea
7, f, CASE RELATIONS, Object:Recipient, an Y receives an X
7, g, CASE RELATIONS, Object:Instrument, a Y is used on an X
7, h, CASE RELATIONS, Recipient:Instrument, Y is an instrument through with X receives some object/service/role
8, a, CAUSE-PURPOSE, Cause:Effect, an X causes Y
8, b, CAUSE-PURPOSE, Cause:Compensatory Action, X causes/compels a person to Y
8, c, CAUSE-PURPOSE, EnablingAgent:Object, X enables the use of Y
8, d, CAUSE-PURPOSE, Action/Activity:Goal, someone/something will X in order to Y
8, e, CAUSE-PURPOSE, Agent:Goal, Y is the goal of X
8, f, CAUSE-PURPOSE, Instrument:Goal, X is intended to produce Y
8, g, CAUSE-PURPOSE, Instrument:Intended Action, Y is the intended action to be taken using X
8, h, CAUSE-PURPOSE, Prevention, X prevents Y
9, a, SPACE-TIME, Item:Location, an X is a place/location/area where Y is found
9, b, SPACE-TIME, Location:Process/Product, an X is a place/location/area where Y is made/done/produced
9, c, SPACE-TIME, Location:Action/Activity, an X is a place/location/area where Y takes place
9, d, SPACE-TIME, Location:Instrument/Associated Item, Y is an instrumental item in the activities that occur at place/location/area Y
9, e, SPACE-TIME, Contiguity, X and Y share a contiguous border
9, f, SPACE-TIME, Time Action/Activity, X is a time when Y occurs
9, g, SPACE-TIME, Time Associated Item, Y is an item associated with time X
9, h, SPACE-TIME, Sequence, an Y follows X in sequence
9, i, SPACE-TIME, Attachment, an X is attached to a Y
10, a, REFERENCE, Sign:Significant, an X indicates/signifies Y
10, b, REFERENCE, Expression, X is an expression that indicates Y
10, c, REFERENCE, Representation, a Y represents/is representative of X
10, d, REFERENCE, Plan, an X is a plan for Y
10, e, REFERENCE, Knowledge, X is the name for knowledge of Y
10, f, REFERENCE, Concealment, X conceals a person/place/thing's Y