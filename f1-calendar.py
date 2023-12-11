import unittest
import math
import csv
import random
from simanneal import Annealer
from deap import base, creator, tools
from math import *

# constants that define the likely hood of two individuals having crossover
# performed and the probability that a child will be mutated. needed for the
# DEAP library
CXPB = 0.5
MUTPB = 0.2

# the unit tests to check that the simulation has been implemented correctly
class UnitTests (unittest.TestCase):
    # this will read in the track locations file and will pick out 5 fields to see if the file has been read correctly
    def testReadCSV(self):
        # read in the locations file
        rows = readCSVFile('track-locations.csv')

        # test that the corners and a middle value are read in correctly
        self.assertEqual('circuit', rows[0][0])
        self.assertEqual('Dec Temp', rows[0][14])
        self.assertEqual('Yas Marina', rows[22][0])
        self.assertEqual('26', rows[22][14])
        self.assertEqual('27', rows[11][8])
    
    # this will test to see if the column conversion works. here we will convert the latitude column and will test 5 values
    # as we are dealing with floating point we will use almost equals rather than a direct equality
    def testColToFloat(self):
        # read in the locations file and convert the latitude column to floats
        rows = readCSVFile('track-locations.csv')
        convertColToFloat(rows, 1)

        # check that 5 of the values have converted correctly
        self.assertAlmostEqual(26.0325, rows[1][1], delta=0.0001)
        self.assertAlmostEqual(24.4672, rows[22][1], delta=0.0001)
        self.assertAlmostEqual(40.3725, rows[4][1], delta=0.0001)
        self.assertAlmostEqual(30.1327, rows[18][1], delta=0.0001)
        self.assertAlmostEqual(25.49, rows[17][1], delta=0.0001)

    # # this will test to see if the column conversion to int works. here we will convert one of the temperature columns and will
    # # test 5 values to see that it worked correctly
    def testColToInt(self):
        # read in the locations file and convert the first of the temperature columns to ints
        rows = readCSVFile('track-locations.csv')
        convertColToInt(rows, 3)

        # check that the values are converted correctly
        self.assertEqual(20, rows[1][3])
        self.assertEqual(24, rows[22][3])
        self.assertEqual(4, rows[11][3])
        self.assertEqual(9, rows[16][3])
        self.assertEqual(23, rows[5][3])
    
    # this will test to see if the file conversion overall is successful for the track locations
    # it will read in the file and will test a string, float, and int from 2 rows to verify it worked correctly
    def testReadTrackLocations(self):
        # read in the locations file
        rows = readTrackLocations()

        # check the name, latitude, and final temp of the first race
        self.assertEqual(rows[0][0], 'Bahrain International Circuit')
        self.assertEqual(rows[0][14],22)
        self.assertAlmostEqual(rows[0][1], 26.0325, delta=0.0001)

        # check the name, longitude, and initial temp of the last race        
        self.assertEqual(rows[21][0], 'Yas Marina')
        self.assertEqual(rows[21][3], 24)
        self.assertAlmostEqual(rows[21][2], 54.603056, delta=0.0001)
    
    # tests to see if the race weekends file is read in correctly
    def testReadRaceWeekends(self):
        # read in the race weekends file
        weekends = readRaceWeekends()

        # check that bahrain is weekend 9 and abu dhabi is weekend 47
        self.assertEqual(weekends[0], 9)
        self.assertEqual(weekends[21], 47)

        # check that hungaroring is weekend 29
        self.assertEqual(weekends[10], 29)
    
    # tests to see if the sundays file is read in correctly
    def testReadSundays(self):
        # read in the sundays file and get the map of sundays back
        sundays = readSundays()

        # check to see the first sunday is january and the last sunday is december
        self.assertEqual(sundays[0], 0)
        self.assertEqual(sundays[51], 11)

        # check a few other random sundays
        self.assertEqual(sundays[10], 2)
        self.assertEqual(sundays[20], 4)
        self.assertEqual(sundays[30], 6)
        self.assertEqual(sundays[40], 9)

    # this will test to see if the haversine function will work correctly we will test 4 sets of locations
    def testHaversine(self):
        # read in the locations file with conversion
        rows = readTrackLocations()

        # check the distance of Bahrain against itself this should be zero
        self.assertAlmostEqual(haversine(rows, 0, 0), 0.0, delta=0.01)
        
        # check the distance of Bahrain against Silverstone this should be 5158.08 km
        self.assertAlmostEqual(haversine(rows, 0, 9), 5158.08, delta=0.01)

        # check the distance of silverstone against monza this should be 1039.49 Km
        self.assertAlmostEqual(haversine(rows, 13, 9), 1039.49, delta=0.01)

        # check the distance of monza to the red bull ring this should be 455.69 Km
        self.assertAlmostEqual(haversine(rows, 13, 8), 455.69, delta=0.01)
    
    # will test to see if the season distance calculation is correct using the 2023 calendar
    def testDistanceCalculation(self):
        # read in the locations & race weekends, generate the weekends, and calculate the season distance
        tracks = readTrackLocations()
        weekends = readRaceWeekends()
        
        # calculate the season distance using silverstone as the home track as this will be the case for 8 of the teams we will use monza
        # for the other two teams.
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 9), 185874.8866, delta=0.0001)
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 13), 179336.2663, delta=0.0001)
    
    # will test that the temperature constraint is working this should fail as azerbijan should fail the test
    def testTempConstraint(self):
        # load in the tracks, race weekends, and the sundays
        tracks = readTrackLocations()
        weekends1 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]
        weekends2 = [9, 11, 43, 30, 37, 21, 40, 34, 22, 35, 29, 26, 27, 24, 44, 42, 46, 18, 38, 13, 17, 47]
        sundays = readSundays()

        # the test with the default calender should be false because of azerbaijan
        self.assertEqual(checkTemperatureConstraint(tracks, weekends1, sundays), False)
        self.assertEqual(checkTemperatureConstraint(tracks, weekends2, sundays), True)
    
    # will test that we can detect four race weekends in a row.
    def testFourRaceInRow(self):
        # weekend patterns the first does not have four in a row the second does
        weekends1 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]
        weekends2 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 41, 42, 43, 44, 46, 47]

        # the first should pass and the second should fail
        self.assertEqual(checkFourRaceInRow(weekends1), False)
        self.assertEqual(checkFourRaceInRow(weekends2), True)
    
    # # will test that we can detect a period for a summer shutdown in july and/or august
    def testSummerShutdown(self):
        # weekend patterns the first has a summer shutdown the second doesn't
        weekends1 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]
        weekends2 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 28, 30, 32, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]

        # the first should pass and the second should fail
        self.assertEqual(checkSummerShutdown(weekends1), True)
        self.assertEqual(checkSummerShutdown(weekends2), False)

# function that will calculate the total distance for the season assuming a given racetrack as the home racetrack
# the following will be assumed:
# - on a weekend where there is no race the team will return home
# - on a weekend in a double or triple header a team will travel straight to the next race and won't go back home
# - the preseason test will always take place in Bahrain
# - for the summer shutdown and off season the team will return home

def calculateSeasonDistance(tracks, weekends, home):
    total_distance = 0.0
    current_location = home
    # Counter for the number of races that have happened
    race_counter = 0  
    # iterate through every week of the year
    for week in range(52):  
        if week in weekends:
            # there is a race this week, calculate the distance from the current location to the race track
            # get the index of the race track in the tracks list
            race_location = race_counter % len(tracks)  
            total_distance += haversine(tracks, current_location, race_location)
            # update the current location to the race track
            current_location = race_location  
            # increment the race counter
            race_counter += 1  
        else:
            # there is no race this week, the team returns home, calculate the distance from the current location to the home track
            total_distance += haversine(tracks, current_location, home)
            # update the current location to the home track
            current_location = home  
    return total_distance




# function that will check to see if there is anywhere in our weekends where four races appear in a row. True indicates that we have four in a row
def checkFourRaceInRow(weekends):
    consecutive_count = 1

    for i in range(1, len(weekends)):
        if weekends[i] - weekends[i - 1] == 1:
            consecutive_count += 1
        else:
            consecutive_count = 1

        if consecutive_count == 4:
            return True

    return False       

# function that will check to see if the temperature constraint for all races is satisfied. The temperature
# constraint is that a minimum temperature of 20 degrees for the month is required for a race to run
def checkTemperatureConstraint(tracks, weekends, sundays):
    temperature_min = 20
    temperature_max = 35

    for i in range(len(weekends)):
        current_weekend = weekends[i]
        current_month = sundays[current_weekend]

        current_track_index = i % len(tracks)
        current_track = tracks[current_track_index]

        current_temperature = current_track[current_month + 3]  # Assuming temperature starts at index 3

        if current_temperature < temperature_min or current_temperature > temperature_max:
            #print(f"Temperature constraint not satisfied for weekend {current_weekend} at track {current_track[0]}")
            #print(f"Temperature: {current_temperature} degrees")
            return False

    return True


# function that will check to see if there is a four week gap anywhere in july and august. we will need this for the summer shutdown.
# the way this is defined is that we have a gap of three weekends between successive races. 
def checkSummerShutdown(weekends):
    consecutive_no_race_weekends = 0

    # Iterate through the weekends
    for week in weekends:
        if week in [26, 27, 29, 30, 34]:  # since 26, 27, 29, 30, 34 are the weeks in July and August
            # If there is no race, increment the consecutive count
            consecutive_no_race_weekends += 1
        else:
            # If there is a race, reset the consecutive count
            consecutive_no_race_weekends = 0

        # Check if there are three successive weekends without a race
        if consecutive_no_race_weekends == 3:
            return True

    return False


# function that will take in the set of rows and will convert the given column index into floating point values
# this assumes the header in the CSV file is still present so it will skip the first row
def convertColToFloat(rows, column_index):
    for row in rows[1:]:  # Skip the header row
        try:
            # Attempt to convert the value at the specified column index to float
            row[column_index] = float(row[column_index])
        except ValueError:
            # Handle the case where the conversion fails (e.g., non-numeric value)
            print(f"Error converting value to float at row {rows.index(row)} and column {column_index}")

        
# funciton that will take in a set of rows and will convert the given column index into integer values
# this assumes the header in the CSV file is still present so it will skip the first row
def convertColToInt(rows, column_index):
    for row in rows[1:]:  # Skip the header row
        try:
            # Attempt to convert the value at the specified column index to int
            row[column_index] = int(row[column_index])
        except ValueError:
            # Handle the case where the conversion fails (e.g., non-numeric value)
            print(f"Error converting value to int at row {rows.index(row)} and column {column_index}")


# function that will use the haversine formula to calculate the distance in Km given two latitude/longitude pairs
# it will take in an index to two rows, and extract the latitude and longitude before the calculation.


def haversine(rows, location1, location2):
    """
    Calculate the distance between two points on the earth's surface.

    Parameters:
    rows (list): List of locations, each location is a list [name, latitude, longitude].
    location1 (int): Index of the first location in the list.
    location2 (int): Index of the second location in the list.

    Returns:
    float: Distance between the two points in kilometers.
    """

    # Radius of the earth in kilometers
    R = 6371.0

    # Coordinates of the two locations
    lat1 = math.radians(rows[location1][1])
    lon1 = math.radians(rows[location1][2])
    if location2 >= len(rows):
     raise ValueError(f"location2 index ({location2}) is out of range for rows of length {len(rows)}")
    if location2 < 0:
        raise ValueError(f"location2 index ({location2}) is negative")

    lat2 = math.radians(rows[location2][1])
    lon2 = math.radians(rows[location2][2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance
    distance = R * c

    return distance

# prints out the itinerary that was generated on a weekend by weekend basis starting from the preaseason test
def printItinerary(tracks, weekends, home, sundays):
    current_location = home
    for week, weekend_num in enumerate(weekends):
        next_weekend_num = weekends[week + 1] if week + 1 < len(weekends) else None

        # Check if the team is currently at home
        if current_location == home:
            if next_weekend_num is not None:
                # Check if the next weekend has a race
                if next_weekend_num == weekends[week] + 1:
                    next_location = tracks[next_weekend_num - 1][0]
                    temp = tracks[next_weekend_num - 1][sundays[week + 1] + 3]
                    print(f"Travelling from home to {next_location}. Race temperature is expected to be {temp} degrees")
                else:
                    print("Staying at home, thus no travel this weekend")
        else:
            if next_weekend_num is not None and next_weekend_num == weekends[week] + 1:
                next_location = tracks[next_weekend_num - 1][0]
                temp = tracks[next_weekend_num - 1][sundays[week + 1] + 3]
                print(f"Travelling directly from {tracks[current_location][0]} to {next_location}. Race temperature is expected to be {temp} degrees")
            else:
                print(f"Travelling home from {tracks[current_location][0]}")

        # Update current location for the next iteration
        current_location = home if next_weekend_num is None else next_location


# function that will take in the given CSV file and will read in its entire contents
# and return a list of lists
def readCSVFile(file):
    # the rows to return
    rows = []

    # open the file for reading and give it to the CSV reader
    csv_file = open(file)
    csv_reader = csv.reader(csv_file, delimiter=',')

    # read in each row and append it to the list of rows.
    for row in csv_reader:
        rows.append(row)

    # close the file when reading is finished
    csv_file.close()

    # return the rows at the end of the function
    return rows

# function that will read in the race weekends file and will perform all necessary conversions on it
def readRaceWeekends():
    # Assuming readCSVFile is defined
    race_weekends_file = 'race-weekends.csv'
    race_weekends_data = readCSVFile(race_weekends_file)
    
    # Extracting and processing race weekends data
    weekends_data = []
    for row in race_weekends_data[1:]:  # Skip header
        weekend_num = int(row[1])
        weekends_data.append(weekend_num)

    return weekends_data


# function that will read in the sundays file that will map the sundays to a list. each sunday maps to a month. we will need this for temperature comparisons later on
def readSundays():
    # Assuming readCSVFile is defined
    sundays_file = 'sundays.csv'
    sundays_data = readCSVFile(sundays_file)
    
    # Extracting and processing Sundays data
    sunday_month_map = {}
    for row in sundays_data[1:]:  # Skip header
        sunday_num = int(row[0])
        month_num = int(row[1])
        sunday_month_map[sunday_num] = month_num

    return sunday_month_map



# function that will read the track locations file and will perform all necessary conversions on it
def readTrackLocations():
    with open('track-locations.csv') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)

        rows = []

        for row in reader:
            # Convert specific columns to float if needed
            row[1] = float(row[1])  # Latitude
            row[2] = float(row[2])  # Longitude
            row[3:] = map(int, row[3:]) # Temperatures

            rows.append(list(row))

        return rows


# function that will run the simulated annealing case for shortening the distance seperately for both silverstone and monza. it will also do a free calendar experiement
# to see if it can be cut down further
def simulated_annealing(initial_state, energy_function, temperature_constraint=None, **kwargs):
    current_state = initial_state
    current_energy = energy_function(current_state, **kwargs)
    best_state = current_state
    best_energy = current_energy

    temperature = 1.0
    cooling_rate = 0.99

    while temperature > 0.1:
        new_state = current_state[:]  # Make a copy of the current state
        # Perform a move to generate a new state, e.g., swap races
        # Implement your move logic here

        new_energy = energy_function(new_state, **kwargs)

        if (
            new_energy < current_energy
            or random.random() < math.exp((current_energy - new_energy) / temperature)
        ):
            current_state = new_state
            current_energy = new_energy

        if current_energy < best_energy:
            best_state = current_state
            best_energy = current_energy

        temperature *= cooling_rate

    return best_state, best_energy

def SACases():
    # Read data from default files
    tracks = readTrackLocations()
    weekends = readRaceWeekends()
    sundays = readSundays()

    # Case 1: Calendar for teams with Silverstone as home track
    print("Simulated Annealing Case 1:")
    initial_state = weekends.copy()
    final_state, final_energy = simulated_annealing(
        initial_state,
        energy_function=calculateSeasonDistance,
        temperature_constraint=checkTemperatureConstraint,
        home_track_index=9  # Index for Silverstone in the tracks list
    )
    printItinerary(tracks, final_state, 9)
    print(f"Total Distance: {final_energy} km")

    # Case 2: Calendar for teams with Monza as home track
    print("\nSimulated Annealing Case 2:")
    initial_state = weekends.copy()
    final_state, final_energy = simulated_annealing(
        initial_state,
        energy_function=calculateSeasonDistance,
        temperature_constraint=checkTemperatureConstraint,
        home_track_index=13  # Index for Monza in the tracks list
    )
    printItinerary(tracks, final_state, 13)
    print(f"Total Distance: {final_energy} km")

    # Case 3: Free calendar
    print("\nSimulated Annealing Case 3:")
    initial_state = weekends.copy()
    final_state, final_energy = simulated_annealing(
        initial_state,
        energy_function=calculateSeasonDistance,
        temperature_constraint=checkTemperatureConstraint,
        free_calendar=True
    )
    printItinerary(tracks, final_state)
    print(f"Total Distance: {final_energy} km")

# Additional Cases...

# You will need to implement the simulated_annealing function separately.


# function that will run the genetic algorithms cases for all four situations
# def GAcases(tracks_file, weekends_file, sundays_file):
#     # Read track locations
#     tracks = readTrackLocations(tracks_file)

#     # Read race weekends
#     weekends = readRaceWeekends(weekends_file)

#     # Read Sundays
#     sundays = readSundays(sundays_file)

#     # Initialize parameters for genetic algorithm
#     population_size = 50
#     generations = 100
#     crossover_rate = 0.8
#     mutation_rate = 0.2

#     # Perform genetic algorithm
#     best_schedule, best_distance = genetic_algorithm(
#         tracks,
#         weekends,
#         home=0,
#         sundays=sundays,
#         population_size=population_size,
#         generations=generations,
#         crossover_rate=crossover_rate,
#         mutation_rate=mutation_rate
#     )

#     # Print the best itinerary and total distance
#     print("Best Itinerary:")
#     printItinerary(tracks, best_schedule, home=0, sundays=sundays)
#     print("Total Distance (Best):", best_distance, "km")

# You will need to implement the genetic_algorithm function separately.



if __name__ == '__main__':
    # uncomment this run all the unit tests. when you have satisfied all the unit tests you will have a working simulation
    # you can then comment this out and move onto your SA and GA solutions
    #unittest.main()

    # run the cases for simulated annealing
    SACases()

    # run the cases for genetic algorithms
    #GAcases()
