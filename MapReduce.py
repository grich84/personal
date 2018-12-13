from mrjob.job import MRJob 
from mrjob.step import MRStep
from datetime import datetime
import sys

# Multi-step jobs
# The map-reduce job here counts the number of reviews per each movie, output is sorted by number of reviews
class MoviesReviewsCount(MRJob):
    def steps(self):
           return [ 
                MRStep(mapper=self.mapper_get_movies,	
                    reducer=self.reducer_count_reviews),
                MRStep(reducer=self.reducer_sortby_print)
                  ]

    #Mapping Function
    def mapper_get_movies(self, _, line):
         (userId, movieId, rating, timestamp) = line.split(',')
         yield movieId, float(rating)

	#Reduce Function
    def reducer_count_reviews(self, key, values):
        listvals = list(values)
        count = float(len(listvals))
        sumval = float(sum(listvals))
        average = float(sumval)/float(count)

        yield '%010f'%float(average), (key, count) 

    ##Output is a string in stdout.. padding zeros 	 
    def reducer_sortby_print(self, ratingAVG,  key):
        for movie in (key):
          yield movie, float(ratingAVG)

if __name__ == '__main__':
    sys.stderr.write("starting your first MapReduce job \n")
    start_time = datetime.now()
    MoviesReviewsCount.run()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
	#Print the time diff in seconds.
    sys.stderr.write("Total execution time "+str(elapsed_time.seconds)+" Seconds\n")