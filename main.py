import sys
from random import random
from math import exp
from time import time


def get_data():  # this is a "middle man" function to build the dictionaries
    file_data, total_participants = read_file()
    tournament_participants = get_participants(file_data, total_participants)
    tournament_weighting = get_weighting(file_data, total_participants)
    return tournament_participants, tournament_weighting


def read_file():  # this is a short function to read the files
    file = open(sys.argv[1])
    file_data = file.read().splitlines()
    total_participants = file_data[0]
    return file_data, total_participants


def get_participants(file_data, total_participants):  # this fills a dictionary with participants from the file provided
    participants = {}
    for i in range(int(total_participants)):  # loop through the data file
        id, name = file_data[i + 1].split(",")
        participants[id] = name
    return participants  # return a dictionary of id->name


def get_weighting(file_data,
                  total_participants):  # this fills a dictionary with participants and weighting from the file provided
    tournament_weighting = {}
    for i in range(int(total_participants) + 2, len(file_data)):  # loop through the file
        weight, participant_A, participant_B = file_data[i].split(",")
        tournament_weighting[(participant_A, participant_B)] = weight
    return tournament_weighting  # return the weighting dictionary


def get_first_random_edge(random_number, ranking_list):  # this function gets the random edge from the ranking
    start_edge, end_edge = 0, 0
    for participant in range(len(ranking_list)):  # loops through the ranking list
        if (int(participant) - 1) / (len(ranking_list) - 1) <= random_number < int(participant) / (
                len(ranking_list) - 1):  # using the random number, determing which edge is selected from the ranking
            start_edge, end_edge = int(participant), int(participant) + 1
    return start_edge, end_edge  # return the nodes to be swapped


def get_random_neighbouring_ranking(current_ranking, tournament_weighting,
                                    cost):  # this function gets the neighbouring solution
    start_edge, end_edge = get_first_random_edge(random(), current_ranking)  # select a random edge
    start = current_ranking[start_edge - 1]
    end = current_ranking[end_edge - 1]
    temp_current_ranking = current_ranking[:]
    temp_current_ranking[start_edge - 1], temp_current_ranking[
        end_edge - 1] = end, start  # swap the nodes on each side of the edge in the ranking
    old_cost = 0
    new_cost = 0
    for matchup, weighting in tournament_weighting.items():  # a loop through the weighting dictionary
        if str(matchup[0]) == str(end) and str(matchup[1]) == str(start):  # calulate the old cost of the two nodes
            old_cost = int(weighting)
        if str(matchup[0]) == str(start) and str(matchup[1]) == str(end):  # calulate the old cost of the two nodes
            new_cost = int(weighting)
    new_cost = (
                       int(cost) - old_cost) + new_cost  # calculate the new cost by subtracting the old cost and adding the new cost
    cost_difference = new_cost - cost
    return temp_current_ranking, new_cost, cost_difference  # return the values


def get_cost(tournament_weighting,
             ranking):  # this function loops through a ranking, and adds up the cost of the weighting that disagrees with the ranking
    ranking_cost = 0
    for matchup, weighting in tournament_weighting.items():
        for i in range(len(ranking)):
            for j in range(i + 1, len(ranking)):
                if str(matchup[0]) == str(ranking[j]) and str(matchup[1]) == str(ranking[i]):
                    ranking_cost = ranking_cost + int(weighting)
    return ranking_cost


def simulated_annealing_algorithm():  # this is the main function which deals with the simulated annealing algorithm
    tournament_participants, tournament_weighting = get_data()  # calls the function to get the participants and the weighting of the participants
    temperature_length = 10
    initial_temperature = 1.0
    current_temperature = initial_temperature
    cooling_ratio = 0.95
    num_non_improve = 8000  # this is high to give us a good kemedy score
    loops_without_optimal_solution = 0
    current_ranking, initial_ranking = list(tournament_participants), list(
        tournament_participants)  # set the inital ranking to the data in numerical order by ID

    cost = get_cost(tournament_weighting, initial_ranking)  # get the initial cost based off the initial ranking
    while loops_without_optimal_solution < num_non_improve:  # outer loop until the condition is met (until num_non_improved is reached)
        for _ in range(temperature_length):  # this is the inner loop of the simulated annealing algorithm
            neighbouring_ranking, new_cost, cost_difference = get_random_neighbouring_ranking(
                current_ranking, tournament_weighting,
                cost)  # this line calls the function that gets the neighbouring ranking and the cost of it
            if cost_difference <= 0:  # if the new solution is better
                current_ranking = neighbouring_ranking[:]  # accept the new ranking as the best ranking
                cost = new_cost  # set the best cost to the lower cost calculated
            else:
                q = random()
                if q < (exp((
                                    -cost) / current_temperature)):  # this accepts the new ranking with the probability equation provided in the SA algorithm
                    current_ranking = neighbouring_ranking[:]
                else:
                    loops_without_optimal_solution += 1  # if the new ranking isn't accepted, add 1 to loops_without_optimal_solution

        current_temperature = current_temperature * cooling_ratio  # multiply the current temperature by the cooling ration to decrease it slightly for the next loop

    print("Rank  |   Name ")  # output the ranks + names of the drivers in a readable/table format
    for i in range(1, len(current_ranking)):
        print(f"{i}   |   {tournament_participants[current_ranking[i]]}")

    print(f"Kemedy Score = {cost}")  # output the total Kenedy Score / cost
    print(float(time() - start_time) * 1000)  # output the total time running in milliseconds


start_time = time()  # get the time the program is executed
simulated_annealing_algorithm()  # this calls the main SA algorithm
