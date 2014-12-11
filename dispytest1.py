# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:55:03 2014

@author: mzhang
"""

# 'compute' is distributed to each node running 'dispynode';
# runs on each processor in each of the nodes
def compute(n):
    import time, socket
    time.sleep(n)
    host = socket.gethostname()
    return (host, n)

if __name__ == '__main__':
    import dispy, random
    cluster = dispy.JobCluster(compute)
    jobs = []
    for n in range(10):
        # run 'compute' with a random number between 5 and 20
        job = cluster.submit(random.randint(5,20))
        job.id = n
        jobs.append(job)
    # cluster.wait() # wait for all scheduled jobs to finish
    for job in jobs:
        host, n = job() # waits for job to finish and returns results
        print('%s executed job %s at %s with %s' % (host, job.id, job.start_time, n))
        # other fields of 'job' that may be useful:
        # print(job.stdout, job.stderr, job.exception, job.ip_addr, job.start_time, job.end_time)
    cluster.stats()
    
    
    