# DM HW3
Q56104076 陳哲緯

# 1. Page rank 

### implement

+ formula




```python
    def pagerank_algorithm(self,iter , d):
        
        page_rank = np.ones( self.n )

        for _ in range(iter):
            new_page_rank = np.zeros_like(page_rank)
            for i in range(self.n):
                # for every node points to me , update page_rank score as sum of (old_page[ni]) / (ni out links)
                for n in self.in_neighbors[i]:
                    new_page_rank[i] += page_rank[n] / len(self.out_neighbors[n])
                
            page_rank = (1-d) * new_page_rank + d/self.n
            
        # norm
        page_rank = page_rank / (page_rank.sum())

        return page_rank
```

## 1-1 Q1



# 2. Hits
# 3. SimRank


