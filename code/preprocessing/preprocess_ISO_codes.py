from mrjob.job import MRJob
from mrjob.step import MRStep
#from mrjob.protocol import RawValueProtocol

considered_locations = [
    "\"India\"", "\"China\"", "\"Iran\"", "\"South Korea\"", "\"South Africa\"", "\"Kenya\"",
    "\"Bangladesh\"", "\"Sweden\"", "\"Norway\"", "\"Germany\"", "\"Italy\"", "\"United Kingdom\"",
    "\"Brazil\"", "\"United States\"", "\"Canada\"", "\"Australia\"", "\"US\"", "\"UK\""
]
airport_types = [
   "\"large_airport\"", "\"small_airport\"", "\"medium_airport\""
]

class joinISOcodes(MRJob):
#    OUTPUT_PROTOCOL = RawValueProtocol

    def steps(self):
        return [
            MRStep(mapper=self.mapper_ISO,
                   reducer=self.reducer_ISO),
            MRStep(mapper=self.mapper_ICAO)
            ]

    def mapper_ISO(self, _, line):
        row = line.split(',')
        # skip first line
        if row[0] != "id":
            # the row is from the ICAO code dataset
            if len(row) >= 9:
                if row[2] in airport_types:
                    yield (row[8], row[1])
            # the row is from country ISO dataset
            elif len(row) > 4 and len(row[0]) == 6:
                yield (row[1], row[2])

    def reducer_ISO(self, key, values):
        yield key, list(values)

    def mapper_ICAO(self, key, codes):
        # ignore empty keys
        if key != "" and len(key) > 2:
            if len(codes) > 1: 
                countryName = ""
                for code in codes:
                    if code in considered_locations:
                        countryName = code
                if countryName != "":
                    for code in codes:
                        if code != countryName:           
                            line = ','.join([code, countryName])
                            yield (None, line)

if __name__ == '__main__':
    joinISOcodes.run()
