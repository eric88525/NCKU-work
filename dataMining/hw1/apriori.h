#include "association.h"

// apriori
vector<assoInfo> apriori(const vector<vector<string>> &data, int minSup)
{
    // return result
    vector<assoInfo> result;

    // lookup table of transations
    vector<map<string, int>> transations;
    // count base item
    map<string, int> oneItemDict;

    // count base item &  save transations for lookup
    for (const vector<string> &t : data)
    {
        map<string, int> tmap;
        for (const auto &item : t)
        {
            oneItemDict[item]++;
            tmap[item]++;
        }

        transations.push_back(tmap);
        // transations.push_back(set<string>(t.begin(), t.end()));
    }

    // one item to add
    set<string> tempBaseItem;
    // add freq one item to baseItems
    for (const auto &item : oneItemDict)
    {
        // one item > minsup
        if (item.second >= minSup)
        {
            tempBaseItem.insert(item.first);
        }
    }

    vector<string> baseItems;
    map<string, int> lastIndex;
    int lastidx = 0;
    for (const auto &s : tempBaseItem)
    {
        baseItems.push_back(s);
        lastIndex[s] = lastidx++;
    }

    // generate rule
    queue<assoInfo> freqItemSet;

    for (string item : baseItems)
    {
        assoInfo asso(set<string>({item}), oneItemDict[item]);
        freqItemSet.push(asso);
        result.push_back(asso);
    }

    while (freqItemSet.size())
    {
        int n = freqItemSet.size();
        while (n--)
        {
            auto it = freqItemSet.front().itemSet.end();
            it--;
            int startIdx = lastIndex[*it] + 1;

            for (int i = startIdx; i < baseItems.size(); i++)
            {
                assoInfo tempAsso = freqItemSet.front();
                tempAsso.support = 0;
                tempAsso.itemSet.insert(baseItems[i]);

                for (const auto &trans : transations)
                {
                    bool ok = true;
                    for (const auto &s : tempAsso.itemSet)
                    {
                        if (!trans.count(s))
                        {
                            ok = false;
                            break;
                        }
                    }
                    if (ok)
                        tempAsso.support++;
                }

                if (tempAsso.support < minSup)
                    continue;
                result.push_back(tempAsso);
                freqItemSet.push(tempAsso);
            }
            freqItemSet.pop();
        }
    }
    return result;
}
