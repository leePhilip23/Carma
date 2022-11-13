class Score:
    def __init__(self, name, age, gender, yr_driven, previous_crash, year_of_car):
        # Save parameters
        self.name = name
        self.age = age
        self.gender = gender
        self.yr_driven = yr_driven
        self.previous_crash = previous_crash
        self.year_of_car = year_of_car

        # Points to be calculated
        self.age_points = 0
        self.gender_points = 0
        self.yr_driven_points = 0
        self.previous_crash_points = 0
        self.year_of_car_points = 0
        self.total_score = 0

        # Calculate points
        self.calc_age()
        self.calc_gender()
        self.calc_yr_driven()
        self.calc_previous_crash()
        self.calc_year_of_car()

    def calc_age(self):
        # Age 16-24
        if 16 <= self.age <= 24:
            self.age_points = 10
            self.age_points /= 100
            self.total_score += self.age_points

        # Age 25-39
        elif 25 <= self.age <= 39:
            self.age_points = 50
            self.age_points /= 100
            self.total_score += self.age_points

        # Age 40-69
        elif 40 <= self.age <= 69:
            self.age_points = 20
            self.age_points /= 100
            self.total_score += self.age_points

        # Age 70+
        elif self.age >= 70:
            self.age_points = 20
            self.age_points /= 100
            self.total_score += self.age_points

    def calc_gender(self):
        # Male
        if self.gender == 'M':
            self.gender_points = 50
            self.gender_points /= 100
            self.total_score += self.gender_points

        # Female
        elif self.gender == 'F':
            self.gender_points = 50
            self.gender_points /= 100
            self.total_score += self.gender_points

    def calc_yr_driven(self):
        # 0-2 years driven
        if 0 <= self.yr_driven <= 2:
            self.yr_driven_points = 10
            self.yr_driven_points /= 100
            self.total_score += self.yr_driven_points

        # 3-7 years driven
        elif 3 <= self.yr_driven <= 7:
            self.yr_driven_points = 20
            self.yr_driven_points /= 100
            self.total_score += self.yr_driven_points

        # 8-11 years driven
        elif 8 <= self.yr_driven <= 11:
            self.yr_driven_points = 30
            self.yr_driven_points /= 100
            self.total_score += self.yr_driven_points

        # 12+ years driven
        elif self.yr_driven >= 12:
            self.yr_driven_points = 40
            self.yr_driven_points /= 100
            self.total_score += self.yr_driven_points

    def calc_previous_crash(self):
        # 0 previous crashes
        if self.previous_crash == 0:
            self.previous_crash_points = 40
            self.previous_crash_points /= 100
            self.total_score += self.previous_crash_points

        # 1 previous crash
        elif self.previous_crash == 1:
            self.previous_crash_points = 30
            self.previous_crash_points /= 100
            self.total_score += self.previous_crash_points

        # 2 previous crashes
        elif self.previous_crash == 2:
            self.previous_crash_points = 20
            self.previous_crash_points /= 100
            self.total_score += self.previous_crash_points

        # 3+ previous crashes
        elif self.previous_crash_points >= 3:
            self.previous_crash_points = 10
            self.previous_crash_points /= 100
            self.total_score += self.previous_crash_points

    def calc_year_of_car(self):
        # Car is 0-5 years old
        if 0 <= self.year_of_car <= 5:
            self.year_of_car_points = 40
            self.year_of_car_points /= 100
            self.total_score += self.year_of_car_points

        # Car is 6-10 years old
        elif 6 <= self.year_of_car <= 10:
            self.year_of_car_points = 30
            self.year_of_car_points /= 100
            self.total_score += self.year_of_car_points

        # Car is 11-15 years old
        elif 11 <= self.year_of_car <= 15:
            self.year_of_car_points = 20
            self.year_of_car_points /= 100
            self.total_score += self.year_of_car_points

        # Car is 16+ years old
        elif self.year_of_car >= 16:
            self.year_of_car_points = 10
            self.year_of_car_points /= 100
            self.total_score += self.year_of_car_points