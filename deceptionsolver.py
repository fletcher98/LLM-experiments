"""
Designed to aid users playing as investigators in the board game
Deception: Murder in Hong Kong. The forensic scientist inputs details such as names, 
roles, weapons, and crime scene clues to build the game state in its entirety. 
The system allows users to add and edit forensic hints while managing the game state
through a JSON file for persistence. A detailed prompt is constructed from the players’ 
inputs and hints, which is then passed to the Gemma model to generate a deduction
regarding the murderer, the weapon, and the clue. This project illustrates how an 
LLM can be integrated into an interactive application to provide reasoning and 
decision support in a game setting.
"""

import os
import sys
import time
import json
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def spinner_stop_event():
    return getattr(spinner_stop_event, "stop", False)

def spinner():
    # A simple spinner to show activity
    for char in "|/-\\":
        if spinner_stop_event():
            break
        sys.stdout.write(f"\rQuerying model... {char}")
        sys.stdout.flush()
        time.sleep(0.1)

class GameState:
    def __init__(self):
        self.players = {}
        self.forensic_scientist = ""
        self.forensic_hints = []
        self.deductions = []
        
    def save(self, filename="game_state.json"):
        with open(filename, 'w') as f:
            json.dump({
                'players': self.players,
                'forensic_scientist': self.forensic_scientist,
                'forensic_hints': self.forensic_hints,
                'deductions': self.deductions
            }, f)
            
    def load(self, filename="game_state.json"):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.players = data['players']
                self.forensic_scientist = data['forensic_scientist']
                self.forensic_hints = data['forensic_hints']
                self.deductions = data.get('deductions', [])
            return True
        return False

def edit_hints(game_state):
    while True:
        print("\nCurrent hints:")
        for i, hint in enumerate(game_state.forensic_hints):
            print(f"{i+1}. Q: {hint['question']} -> A: {hint['answer']}")
        
        action = input("\nEnter hint number to edit, 'A' to add, or 'Q' to quit: ").strip().lower()
        if action == 'q':
            break
        elif action == 'a':
            question = input("New question: ").strip()
            answer = input("New answer: ").strip()
            if question and answer:
                game_state.forensic_hints.append({"question": question, "answer": answer})
        elif action.isdigit():
            idx = int(action) - 1
            if 0 <= idx < len(game_state.forensic_hints):
                question = input(f"Edit question [{game_state.forensic_hints[idx]['question']}]: ").strip()
                answer = input(f"Edit answer [{game_state.forensic_hints[idx]['answer']}]: ").strip()
                if question:
                    game_state.forensic_hints[idx]['question'] = question
                if answer:
                    game_state.forensic_hints[idx]['answer'] = answer

def input_weapons(player_num, weapons):
    print("Enter the four weapons for this player (type 'U' to undo the last entry):")
    while len(weapons) < 4:
        w = input(f"- Weapon {len(weapons) + 1}: ").strip()
        if w.lower() == 'u':
            if weapons:
                removed_weapon = weapons.pop()
                print(f"Undid last weapon entry: {removed_weapon}")
            else:
                print("Nothing to undo in weapons!")
        elif w:
            weapons.append(w)
        else:
            print("Weapon cannot be empty. Please enter a valid weapon.")

def input_clues(player_num, clues, weapons):
    print("Enter the four crime scene clues for this player (type 'U' to undo the last entry):")
    while len(clues) < 4:
        c = input(f"- Clue {len(clues) + 1}: ").strip()
        if c.lower() == 'u':
            if clues:
                removed_clue = clues.pop()
                print(f"Undid last clue entry: {removed_clue}")
            elif weapons:
                removed_weapon = weapons.pop()
                print(f"Undid last weapon entry: {removed_weapon}")
                return 'undo_weapon'
            else:
                print("Nothing to undo in clues or weapons!")
        elif c:
            clues.append(c)
        else:
            print("Clue cannot be empty. Please enter a valid clue.")        

def setup_game():
    try:
        player_count = int(input("How many players are in the game (4-12)? ").strip())
        if not 4 <= player_count <= 12:
            raise ValueError("Player count must be between 4 and 12.")
    except ValueError as ve:
        print(f"Invalid input: {ve}")
        sys.exit(1)

    forensic_scientist = input("Enter the name of the Forensic Scientist: ").strip()
    has_accomplice = player_count >= 6 and input("Include Accomplice role? (y/n): ").lower() == 'y'
    has_witness = player_count >= 7 and input("Include Witness role? (y/n): ").lower() == 'y'

    return {
        "player_count": player_count,
        "forensic_scientist": forensic_scientist,
        "has_accomplice": has_accomplice,
        "has_witness": has_witness
    }

def main():
    hf_token = 
    model_name = 

    print("Loading the model, this may take a while...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            use_auth_token=hf_token, 
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create a text-generation pipeline
    try:
        llm = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_length=1024, 
            temperature=0.7, 
            top_p=0.9, 
            repetition_penalty=1.1,
            device_map="auto",
            truncation=True  # Explicitly set truncation
        )
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        sys.exit(1)

    game_state = GameState()
    if len(sys.argv) > 1 and sys.argv[1] == '--continue':
        if not game_state.load():
            print("No saved game found, starting new game...")
    
    if not game_state.players:  # New game
        game_setup = setup_game()
        game_state.forensic_scientist = game_setup['forensic_scientist']
        remaining_players = game_setup['player_count'] - 1
        
        print(f"\nForensic Scientist: {game_state.forensic_scientist}")
        
        for i in range(1, remaining_players + 1):
            player_name = input(f"\nEnter name for Player {i}: ").strip() or f"Player{i}"
            print(f"\nEntering data for {player_name}:")
            weapons = []
            clues = []
            
            while True:
                # Input Weapons (Means of Murder)
                print("Enter the four Means of Murder cards for this player (type 'U' to undo):")
                while len(weapons) < 4:
                    w = input(f"- Means {len(weapons) + 1}: ").strip()
                    if w.lower() == 'u':
                        if weapons:
                            removed_weapon = weapons.pop()
                            print(f"Undid last means entry: {removed_weapon}")
                        else:
                            print("Nothing to undo!")
                    elif w:
                        weapons.append(w)
                    else:
                        print("Card cannot be empty. Please enter a valid card.")

                # Input Clues (Scene Evidence)
                clues_result = input_clues(i, clues, weapons)
                if clues_result == 'undo_weapon':
                    continue
                else:
                    break

            role = "Investigator"  # Default role
            if i == 1:  # First player input is assumed to be the Murderer for simplicity
                role = "Murderer"
            elif game_setup['has_accomplice'] and i == 2:
                role = "Accomplice"
            elif game_setup['has_witness'] and i == 3:
                role = "Witness"

            game_state.players[player_name] = {
                "role": role,
                "weapons": weapons,
                "clues": clues
            }

    def construct_prompt(game_state):
        players_info = []
        for player, assets in game_state.players.items():
            p_info = (
                f"{player} ({assets['role']}):\n"
                f"  Weapons: {', '.join(assets['weapons'])}\n"
                f"  Clues: {', '.join(assets['clues'])}\n"
            )
            players_info.append(p_info)
        players_str = "\n".join(players_info)
        
        if game_state.forensic_hints:
            hints_str = "\n".join([f"Round {i+1}: {h['question']} -> {h['answer']}" for i,h in enumerate(game_state.forensic_hints)])
        else:
            hints_str = "No hints yet."

        prompt = (
            "You are analyzing a game of 'Deception: Murder in Hong Kong'. The objective is to deduce which player is the murderer and which weapon and clue were chosen by the murderer.\n\n"
            "Below is the current game state and the forensic hints.\n\n"
            "GAME STATE:\n"
            f"{players_str}\n\n"
            "HINTS:\n"
            f"{hints_str}\n\n"
            "You are a reasoning assistant. Your task: Analyze the hints and game state.\n"
            "DO NOT repeat the game state or hints in your answer. DO NOT restate the entire prompt.\n"
            "INSTEAD, directly explain your reasoning process and then provide your best guess at the murderer, weapon, and clue.\n"
            "Finally, rate your confidence.\n\n"
            "Now, please provide your reasoning and conclusion:"
        )
        
        return prompt

    while True:
        print("\nOptions:")
        print("1. Add/Edit forensic hints")
        print("2. Get deduction")
        print("3. Review previous deductions")
        print("4. Save game")
        print("5. Exit")
        
        choice = input("Choose an option: ").strip()
        
        if choice == '1':
            edit_hints(game_state)
        elif choice == '2':
            prompt = construct_prompt(game_state)

            # Start a spinner in a separate thread to indicate that we're waiting for the response
            spinner_stop_event.stop = False
            spin_thread = threading.Thread(target=spinner)
            spin_thread.start()

            try:
                # Query the LLM
                response = llm(prompt, num_return_sequences=1, do_sample=True)
            except Exception as e:
                print(f"\nError querying the model: {e}")
                spinner_stop_event.stop = True
                spin_thread.join()
                continue

            # Stop the spinner
            spinner_stop_event.stop = True
            spin_thread.join()
            # Clear the spinner line by printing a new line
            print()

            # Get the generated text
            response_text = response[0]['generated_text'].strip()

            # Remove the prompt content from the response
            if response_text.startswith(prompt):
                cleaned_response = response_text[len(prompt):].strip()
            else:
                cleaned_response = response_text  # Fallback in case it doesn't match exactly

            # Optionally, you can further clean the response if needed
            # For example, remove any leading dashes or other artifacts
            # cleaned_response = cleaned_response.lstrip("- ").strip()

            # Print the final processed response
            print("\nLLM's Deduction:")
            print(cleaned_response)

            game_state.deductions.append({
                'round': len(game_state.forensic_hints),
                'deduction': cleaned_response
            })
        elif choice == '3':
            print("\nPrevious deductions:")
            for d in game_state.deductions:
                print(f"\nRound {d['round']}:")
                print(d['deduction'])
        elif choice == '4':
            game_state.save()
            print("Game saved!")
        elif choice == '5':
            break

if __name__ == "__main__":
    main()