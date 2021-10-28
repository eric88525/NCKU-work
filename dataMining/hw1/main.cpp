#include "fpGrowth.h"

// get all transations
vector<vector<string>> readData(string filePath)
{
    int id, tranID;
    string item;

    ifstream infile(filePath);

    map<int, vector<string>> datas;

    while (infile >> id >> tranID >> item)
    {
        datas[tranID].push_back(item);
    }

    vector<vector<string>> result;

    for (auto &d : datas)
    {
        result.push_back(d.second);
    }
    return result;
}

// given itemset { 1, 2, 3}
// return all combination  of {} => {} ex. like {1}=>{2,3}
vector<pair<string, string>> allCombination(const set<string> &freqItemSet)
{
    vector<pair<string, string>> result;

    queue<pair<string, string>> que;

    vector<string> itemSet(freqItemSet.begin(), freqItemSet.end());

    que.push(make_pair("", ""));

    int index = 0;

    while (que.size() && index < itemSet.size())
    {

        int n = que.size();
        while (n--)
        {
            auto temp1 = que.front();
            auto temp2 = que.front();
            que.pop();

            temp1.first += itemSet[index] + ", ";
            temp2.second += itemSet[index] + ", ";

            que.push(temp1);
            que.push(temp2);
        }
        index++;
    }
    while (que.size())
    {
        if (que.front().first != "" && que.front().second != "")
            result.push_back(que.front());
        que.pop();
    }
    return result;
}

// output result to file
void printResult(map<string, int> itemFrequency, const vector<assoInfo> &freqSet, string outfile, float minSupport, float minConfidence, float transCount)
{
    ofstream myfile;
    myfile.open(outfile);

    int totalCounts = 0;

    map<string, float> ruleCount;

    ofstream rulefile;
    rulefile.open("./rule.txt");

    // create itemset:freq map
    for (auto const &fSet : freqSet)
    {
        string k = "";
        for (auto const &item : fSet.itemSet)
            k += item + ", ";

        ruleCount[k] = fSet.support;
        rulefile << k << " | " << fSet.support << "\n";
    }
    rulefile.close();

    myfile << " Sup  | Conf  | Rule\n";
    myfile << "====================\n";

    //  confidence filter & print reuslt to file
    for (auto const &itSet : freqSet)
    {
        auto allCombs = allCombination(itSet.itemSet);

        for (auto const &comb : allCombs)
        {

            float sup = float(itSet.support) / transCount;
            float confi = float(itSet.support) / ruleCount[comb.first];

            if (confi < minConfidence)
                continue;

            myfile << setprecision(4) << setw(5) << sup << " | " << setw(5) << confi << " | ";
            myfile << "{ " << comb.first << "} => { " << comb.second << "}\n";

            totalCounts++;
        }
    }
    myfile << "====================\nTotal rules = " << totalCounts << endl;
    myfile.close();
}

int main()
{

    float minSupport = 0.1, confidence = 0.2;

    // read data
    vector<vector<string>> datas = readData("./data.txt");

    /*  vector<vector<string>> datas = {
        {"milk", "bread", "beer"},
        {"bread", "coffee"},
        {"bread", "egg"},
        {"milk", "bread", "coffee"},
        {"milk", "egg"},
        {"bread", "egg"},
        {"milk", "egg"},
        {"milk", "bread", "egg", "beer"},
        {"milk", "bread", "egg"},
    };*/
    // create tree
    //fpTree tree(minSupport * datas.size());
    fpTree tree(int(datas.size() * minSupport));
    // build tree
    tree.buildTree(datas);

    auto tree_ans = tree.fpMining();

    printResult(tree.frequency, tree_ans, "./fp_result.txt", minSupport, confidence, datas.size());
    int x = 0;
}