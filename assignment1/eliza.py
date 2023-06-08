"""
# AIT 526 - Natural Language Processing
# Assignment  1: Programming Assignment – Chatbot Eliza
# Group 9:
# Yasser Parambathkandy
# Indranil Pal
# 6/7/2023
"""

"""
Problem Description:
Eliza is a program that simulates conversation with a human user using simple pattern matching and transformation rules.
The goal of Eliza was not to provide intelligent or deep understanding of human language, but rather to mimic
a psychotherapist. The focus is on active listening and reflection, allowing the client to explore their thoughts and
feelings. Eliza emulates this approach by posing questions and making statements that encourage the user to elaborate
on their concerns.  

Setup and usage Instructions:
1. Install Python 3.9.0
2. Install nltk 
2. Run the program using python3 eliza.py
3. Enter the text you wish to chat with Eliza, the psychotherapist
4. Enter 'quit' if you wish to discontinue

Example program input and output:
eliza>> Hello there, My name is Eliza, your psychotherapist. What is your name?
user>> Robert
eliza>> Hello Robert, How are you feeling today?
Robert>>  I am feeling sad
eliza>> Why are you feeling sad?
Robert>>  I had a panic attack yesterday
eliza>> Tell me more about the panic attack yesterday.
Robert>>  I don't like crowded places
eliza>> Why don't you like crowded places?
Robert>>  I feel happy when I'm with my friends
eliza>> What is contributing to your feelings of happy when you are with your friends?
Robert>>  I am constantly worried about the future
eliza>> Why are you constantly worried about the future?
Robert>>  I had a nightmare last night
eliza>> Tell me more about the nightmare last night.
Robert>>  I don't enjoy socializing
eliza>> Tell me more
Robert>>  I feel lonely most of the time
eliza>> What is contributing to your feelings of lonely most of the time?
Robert>>  I am experiencing a lot of stress at work
eliza>> Why are you experiencing a lot of stress at work?
Robert>>  I get angry easily
eliza>> What situations typically trigger your anger?
Robert>>  I feel exhausted all the time
eliza>> What is contributing to your feelings of exhausted all the time?
Robert>>  I had a disagreement with my partner
eliza>> Tell me more about the disagreement with your partner.
Robert>>  I feel lost and confused about my life
eliza>> What is contributing to your feelings of lost and confused about your life?
Robert>>  I struggle with low self-esteem
eliza>> What are some factors that contribute to your feelings of low self-esteem?
Robert>>  I am addicted to smoking
eliza>> Why are you addicted to smoking?
Robert>>  quit
It was a pleasure talking to you

Bonus Functionality:
1) Validate name of user in the first response and if it is invalid, prompt again for a valid name.
2) If the user inputs text that is not in the response dictionary for three consecutive times, the program will
   provide a response that it couldn't understand the user and will refer to a senior psychotherapist.

Algorithm:
This code utilizes a rule-based algorithm using regular expressions, where predefined answers are determined based 
on the user's input. The program is implemented as below:
1) Print an introduction message and ask the user for their name.
2) Validate and extract the first name from the user's response.
3) Check if the user entered the quit command. If so, terminate the conversation.
4) Set a flag is_first_question to True to track if it is the first question.
5) Initialize an empty list called last_questions to store the last three questions asked.
6) Enter a while loop until the quit command is entered.
7) Check if the user's response is blank. If so, prompt the user to provide a response and continue to the next iteration of the loop.
8) Check if the user's name is empty or contains no alphabetic characters. 
If so, prompt the user to provide a valid name and continue to the next iteration of the loop.
9) If it is the first question, ask the patient how they are feeling today. Set is_first_question to False.
10) If it is not the first question, generate a new question using the generate_question_to_user function based on the 
patient's response. Append the new question to last_questions.
11) Check if last_questions contains three questions. If so, remove the oldest question from the list.
12) Prompt the patient for their response to the generated question.
13) Check if last_questions contains three questions and all of them are "Tell me more". If so, print a fallback message and break out of the loop.
14) Check if the patient's response matches the quit command. If so, break out of the loop.
15) Continue the loop by generating a new question based on the patient's response.
16) Repeat steps 7-15 until the quit command is entered.
17) Print a closing message: "It was a pleasure talking to you."
"""

import re

from nltk.tokenize import word_tokenize

# all possible user response and the next question to ask
predict_responses = {
    r'[Hh](ello|i).*': 'Hey there! How are you feeling today?',
    r'[Ii] am feeling (.*)': "Why are you feeling {0}?",
    r'[Ii] am (.*)': "Why are you {0}?",
    r'[Ii] had an?(.*)': "Tell me more about the {0}.",
    r'^[Ii] do not like (.+)': "Why don't you like {0}?",
    r'[Ii] feel (.*)(.*)': "What is contributing to your feelings of {0}?",
    r'^[Ii] don\'?t feel (.*)': "What do you think is causing your lack of {0}?",
    r'(.*)[Ss]ad(.*)': "What events or thoughts have led to your sadness?",
    r'[Ii] am anxious about (.*)': "Why does {0} make you anxious?",
    r'[Ii] had a panic attack because (.*)': "Tell me more about the situation that triggered your panic attack.",
    r'^[Ii] don\'?t like (.*)': "What is it about {0} that you don't like?",
    r'(.*)[Oo]verwhelm(.*)': "What factors are overwhelming you?",
    r'(.*)[Ww]orried about the future(.*)': "What specific concerns do you have about the future?",
    r'(.*)[Nn]ightmare(.*)': "Can you describe the details of the nightmare?",
    r'^[Ii] don\'?t enjoy (.*)': "What is it about {0} that you don't enjoy?",
    r'(.*)[Ll]onely(.*)': "When do you tend to feel the most lonely?",
    r'(.*)[Ss]tress at work(.*)': "What aspects of your work are causing you stress?",
    r'(.*)[Aa]ngry(.*)': "What situations typically trigger your anger?",
    r'(.*)[Ee]xhausted(.*)': "Have you noticed any patterns or specific reasons behind your constant exhaustion?",
    r'(.*)[Dd]isagreement with my partner(.*)': "What was the disagreement about?",
    r'(.*)[Ll]ost and confused about my life(.*)': "In what areas of your life do you feel the most lost and confused?",
    r'(.*)[Ll]ow self-esteem(.*)': "What are some factors that contribute to your feelings of low self-esteem?",
    r'(.*)[Aa]ddicted to smoking(.*)': "How often and when do you find yourself smoking the most?",
    r'(.*)': "Tell me more"  # the default question on no match

}

pronouns = {
    "you": "me",
    "me": "you",
    "your": "my",
    "my": "your",
    "are": "am",
    "am": "are",
    "i": "you"
}


def clean_response(user_response):
    """
    Cleans up contractions and punctuation in the user's response.
    Args:
        user_response (str): The user's response to clean up.
    Returns:
        str: The cleaned up user's response.
    """
    # Define a pattern to match contractions like "I'd" or "I'd an"
    had_pattern = r'[Ii]\'d an?'
    user_response = str(user_response)

    # Substitute contractions with their expanded forms
    user_response = re.sub(r'[Ii]\'m', "I am", user_response)

    # If the user's response matches the had_pattern, replace "I'd" with "I had",
    # otherwise replace "I'd" with "I would"
    if re.match(had_pattern, user_response):
        user_response = re.sub(r'[Ii]\'d', "I had", user_response)
    else:
        user_response = re.sub(r'[Ii]\'d', r"I would", user_response)

    user_response = re.sub(r"[Ii]t\'s", "It is", user_response)
    user_response = re.sub(r'[Ii]\'ve', "I have", user_response)
    user_response = re.sub(r'[Ww]hat\'s', "What is", user_response)
    user_response = re.sub(r'[Ll]et\'s', "Let us", user_response)
    user_response = re.sub(r'[Dd]on\'t', "do not", user_response)

    return user_response


def replace_first_person_pronouns(sentence):
    """
    Replaces first-person pronouns with second-person pronouns in a given sentence.
    Args:
        sentence (str): The sentence in which pronouns need to be replaced.
    Returns:
        str: The sentence with first-person pronouns replaced by second-person pronouns.
    """
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)

    # Loops through all the words and replaces the first-person pronoun with second-person pronoun
    # i is the index and token is the word at that index
    for i, token in enumerate(tokens):
        if token in pronouns:
            tokens[i] = pronouns[token]

    # Returns the modified sentence with pronouns replaced
    return ' '.join(tokens)


def generate_question_to_user(user_response):
    """
    Generates a question based on the user's response.
    Args:
        user_response (str): The user's response to generate a question from.
    Returns:
        str: The generated question based on the user's response.
    """
    user_response = clean_response(user_response)
    for user_response_pattern, new_question in predict_responses.items():
        # Checks whether the user's response matches any patterns
        match = re.match(user_response_pattern, user_response)
        if match:
            # Extracts the words after the matching pattern is located in the sentence
            s = ""
            for x in match.groups():
                s = s + " " + x

            # Replaces first-person pronouns with second-person pronouns in the extracted words
            replaced_words = replace_first_person_pronouns(s)
            # Replaces the corresponding placeholders in the new_question with the replaced words
            response_message = new_question.format(*[replaced_words])
            return response_message


def validate_and_extract_name(name):
    """
    Validates and extracts the first name from the given name.
    Args:
        name (str): The name to validate and extract the first name from.
    Returns:
        str: The extracted first name, or an empty string if the name is invalid.
    """
    # Search for a pattern that matches "my name is" or "I am" followed by a name
    match = re.search(r'(?:my name is|I am)?\s*([a-zA-Z]+(?: [a-zA-Z]+)?)', name)
    # Initialize the first_name variable
    first_name = ''
    # Check if the name is empty or doesn't contain any alphabetic characters
    if not name or not re.search(r'[a-zA-Z]', name):
        first_name = ''
    # If a match is found, extract the first name from the match group
    elif match:
        first_name = match.group(1)
    return first_name


def conversation():
    """
    Conducts a conversation with the user using a psychotherapist-like dialogue.
    Returns:
        None
    """
    intro = "Hello! Welcome to AIT526 office of psychotherapist\n"
    intro += "Please type 'quit' to stop the session\n"
    print(intro)
    # Prompt the user for their name
    user_response = input("eliza>> Hello there, My name is Eliza, your psychotherapist. What is your name?\nuser>> ")
    # Extract the first name from the user's response
    first_name = validate_and_extract_name(user_response)
    quit_command_regex = r'[Qq]uit;?'
    command_quit = re.match(quit_command_regex, first_name)
    is_first_question = True
    last_questions = []

    while not command_quit:
        # Check if the user entered a blank input
        if user_response == "":
            user_response = input("eliza>> Please provide a response.\nuser>> ")
            continue
        elif not first_name:
            user_response = input("eliza>> Please provide your valid name.\nuser>> ")
            first_name = validate_and_extract_name(user_response)
        elif is_first_question:
            user_response = input(f"eliza>> Hello {first_name}, How are you feeling today?\n{first_name}>>  ")
            is_first_question = False
        else:
            # Generate a new question based on the user's response
            new_question = generate_question_to_user(user_response)
            last_questions.append(new_question)
            # Keep track of the last 3 questions asked
            if len(last_questions) > 3:
                last_questions.pop(0)
            user_response = input(f"eliza>> {new_question}\n{first_name}>>  ")
            # Check if the conversation is stuck in a loop and unable to make progress
            if len(last_questions) == 3 and all(question == "Tell me more" for question in last_questions):
                print(
                    "Sorry, I am unable to help you further. Please email ait526@gmu.edu to talk to an advanced technician. Exiting...")
                break
        # Check if the user entered the quit command
        command_quit = re.match(quit_command_regex, user_response)
    print("It was a pleasure talking to you")


if __name__ == '__main__':
    conversation()
