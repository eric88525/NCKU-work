#include "association.h"

// apriori
vector<assoInfo> apriori(const vector<vector<string>> &data, float minSup)
{
    // return result
    vector<assoInfo> result;

    // dictionary format transations, for example:
    // [{"apple":1, "banana":1, "orange":1}, {"apple":1, "banana":1, "orange":1}]
    vector<map<string, int>> transations;
    // count base item
    map<string, int> itemCounts;

    // count base item &  save transations for lookup
    for (const vector<string> &t : data)
    {
        map<string, int> tmap;
        for (const auto &item : t)
        {
            itemCounts[item]++;
            tmap[item]++;
        }
        transations.push_back(tmap);
        // transations.push_back(set<string>(t.begin(), t.end()));
    }

    vector<string> baseItems;
    int lastidx = 0;

    // the index of item in baseItems, for example:
    // baseItems = ["apple", "banana", "orange"]
    // itemIndex = {"apple":0, "banana":1, "orange":2}
    map<string, int> itemIndex;

    // add freq one item to baseItems
    for (const auto &item : itemCounts)
    {
        if (item.second >= minSup)
        {
            baseItems.push_back(item.first);
            itemIndex[item.first] = lastidx++;
        }
    }

    // generate rule
    queue<assoInfo> itemSet;

    for (string item : baseItems)
    {
        assoInfo asso(set<string>({item}), itemCounts[item]);
        itemSet.push(asso);
        result.push_back(asso); // add one frequent itemset to result
    }

    while (itemSet.size())
    {
        int n = itemSet.size();
        while (n--)
        {
            // get last item in itemset
            auto it = itemSet.front().itemSet.end();
            it--;

            for (int i = itemIndex[*it] + 1; i < baseItems.size(); i++)
            {
                assoInfo items = itemSet.front();

                items.appearCount = 0;
                items.itemSet.insert(baseItems[i]);

                for (const auto &trans : transations)
                {
                    bool notInTrans = false;
                    // loop through all the item in itemSet
                    for (const auto &s : items.itemSet)
                    {
                        if (!trans.count(s)) // tras not contain item
                        {
                            notInTrans = true;
                            break;
                        }
                    }
                    if (!notInTrans)
                        items.appearCount++;
                }

                if (items.appearCount >= minSup)
                {
                    result.push_back(items);
                    itemSet.push(items);
                }
            }
            itemSet.pop();
        }
    }
    return result;
}
