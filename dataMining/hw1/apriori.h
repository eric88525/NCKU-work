#include "association.h"

// apriori
vector<assoInfo> apriori(const vector<vector<string>> &data, int minSup)
{
    // return result
    vector<assoInfo> result;

    // lookup table of transations
    vector<set<string>> transations;

    // count base item
    map<string, int> oneItemDict;

    // save seen freq itemset
    map<string, int> itemSetHistory;

    // count base item &  save transations
    for (vector<string> const &t : data)
    {
        set<string> temp;
        for (const auto &item : t)
        {
            oneItemDict[item]++;
            temp.insert(item);
        }
        transations.push_back(temp);
    }
    // one item to add
    vector<string> baseItems;
    vector<assoInfo> freqItemSet;

    // add freq one item to baseItems
    for (const auto &item : oneItemDict)
    {
        if (item.second >= minSup)
        {
            baseItems.push_back(item.first);
            assoInfo temp(set<string>({item.first}), item.second);
            freqItemSet.push_back(temp);
            result.push_back(temp);
        }
    }
    while (freqItemSet.size())
    {
        // next step freq item set
        vector<assoInfo> currItemSet;

        for (assoInfo const &f : freqItemSet)
        {
            // base item to add
            for (int i = 0; i < baseItems.size(); i++)
            {
                string oItem = baseItems[i];

                // if already has base item
                if (f.itemSet.count(oItem))
                    continue;

                // add base item
                set<string> tempSet = f.itemSet;
                tempSet.insert(oItem);

                // check if this pattern exist
                string pattern = "";
                for (auto const &i : tempSet)
                    pattern += i + ", ";

                if (itemSetHistory.count(pattern))
                    continue;
                else
                    itemSetHistory[pattern] = 1;

                assoInfo tempAsso(tempSet, 0);

                // lookup support from transations
                for (set<string> const &trans : transations)
                {
                    // get union of this freq itemset with transation
                    set<string> ts;
                    // if intersection count == this pattern means it show in transation
                    set_intersection(tempAsso.itemSet.begin(), tempAsso.itemSet.end(), trans.begin(), trans.end(), inserter(ts, ts.begin()));

                    if (ts.size() == tempAsso.itemSet.size())
                    {
                        tempAsso.support++;
                    }
                }
                // delete if support < minup
                if (tempAsso.support < minSup)
                    continue;
                else
                {
                    currItemSet.push_back(tempAsso);
                    result.push_back(tempAsso);
                }
            }
        }
        freqItemSet = currItemSet;
    }

    return result;
}
