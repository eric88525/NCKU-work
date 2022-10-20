#include "fpGrowth.h"
#include "apriori.h"

// get all transations
vector<vector<string>> readIBMData(string filePath)
{

    cout << "==========================================\n";
    cout << "IBM Dataset" << endl << filePath << endl;
    cout << "==========================================\n";
    string id, tranID, item;
    ifstream infile(filePath);
    map<string, vector<string>> datas;
    vector<vector<string>> result;

    while (infile >> id >> tranID >> item)
        datas[tranID].push_back(item);

    for (auto &d : datas)
        result.push_back(d.second);

    return result;
}

vector<vector<string>> readKaggleData(string filepath)
{

    string inp;
    ifstream infile(filepath);

    map<string, vector<string>> datas;

    while (getline(infile, inp))
        datas[inp.substr(0, inp.find(','))].push_back(inp.substr(inp.find(',') + 1));

    vector<vector<string>> result;
    for (auto &d : datas)
        result.push_back(d.second);

    infile.close();
    return result;
}

// print result to file
void resultToFile(vector<assoInfo> &freqSet, string freqPath, string rulePath, float minSupport, float minConfidence, float transCount)
{

    int totalRules = 0;

    map<string, float> ruleCount;

    ofstream freqFile;
    freqFile.open(freqPath);

    sort(freqSet.begin(), freqSet.end(), [](assoInfo &a, assoInfo &b)
         { return a.appearCount > b.appearCount; });

    cout << "[freqSet] = " << freqSet.size() << "\n";
    // create itemset:freq map
    freqFile << "count | itemSet\n";
    for (const auto &fSet : freqSet)
    {
        string k = "";
        for (const auto &item : fSet.itemSet)
            k += item + ", ";

        ruleCount[k] += fSet.appearCount;

        freqFile << setw(5) << setprecision(4) << fSet.appearCount << " | { " << k << "}\n";
    }
    freqFile.close();

    ofstream ruleFile;
    ruleFile.open(rulePath);
    ruleFile << " Sup  | Conf  | Rule\n";
    ruleFile << "====================\n";

    //  confidence filter & print reuslt to file
    for (const auto &itSet : freqSet)
    {
        auto allCombs = allCombination(itSet.itemSet);

        for (const auto &comb : allCombs)
        {

            string leftRule = "";
            for (const auto &item : comb.first)
                leftRule += item + ", ";

            string rightRule = "";
            for (const auto &item : comb.second)
                rightRule += item + ", ";

            float sup = float(itSet.appearCount) / float(transCount);
            float confi = float(itSet.appearCount) / float(ruleCount[leftRule]);

            if (confi > 1)
                cout << "wrong\n";

            if (confi < minConfidence)
                continue;

            ruleFile << setprecision(5) << setw(5) << sup << " | " << setprecision(5) << setw(5) << confi << " | ";
            ruleFile << "{ " << leftRule << "} => { " << rightRule << "}\n";

            totalRules++;
        }
    }
    ruleFile << "====================\nTotal rules = " << totalRules << endl;
    ruleFile.close();
    cout << "[rules] = " << totalRules << "\n";
}

void test(string mode, const vector<vector<string>> &datas, float minSupport = 0.1, float confidence = 0.2)
{
    float mSup = datas.size() * minSupport;
    vector<assoInfo> freqItemSet;

    cout << "[mode] = " << mode << "\n";
    cout << "[transations] = " << datas.size() << "\n";
    cout << "[minSup] = " << mSup << "\n";
    cout << "[minConf] = " << confidence << "\n";

    // count time
    auto start = chrono::steady_clock::now();

    if (mode == "fp")
    {
        fpTree tree(mSup);
        tree.buildTree(datas);
        freqItemSet = tree.fpMining(assoInfo());
    }
    else if (mode == "ap")
    {
        freqItemSet = apriori(datas, mSup);
    }
    else
    {
        cout << "mode error\n";
        return;
    }

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << "[run time] = " << chrono::duration<double, milli>(diff).count() << " ms" << endl;
    cout << ">>>>>>\n";
    if (mode == "fp")
    {
        resultToFile(freqItemSet, "./fp_result.txt", "./fp_rule.txt", minSupport, confidence, datas.size());
    }
    else if (mode == "ap")
    {
        resultToFile(freqItemSet, "./ap_result.txt", "./ap_rule.txt", minSupport, confidence, datas.size());
    }
    cout << "=====================\n";
}

int main()
{
    /*
        test (mode,datas,minSupport,confidence)
        mode: ( fp or ap )
        data: transation data
        support: min support
        confidence: min confidence
    */

    //test on ibm data
    // auto ibmData = readIBMData("./dataset/IBM2021.txt");
    // test("fp", ibmData, 0.01, 0.8);
    // test("ap", ibmData, 0.01, 0.8);

    auto ibmData = readIBMData("./dataset/IBM5000.txt");
    test("fp", ibmData, 0.01, 0.8);
    test("ap", ibmData, 0.01, 0.8);

    // auto testData = readIBMData("./dataset/toy.txt");
    // test("fp", testData, 0.5, 0.8);
    // test("ap", testData, 0.5, 0.8);

    // auto testData = readIBMData("./dataset/dm_test.txt");
    // test("fp", testData, 0.1, 0.1);
    // test("ap", testData, 0.1, 0.1);

    // test("ap", ibmData, 0.01, 0.8);

    // test on kaggle data

    // auto kaggleData = readKaggleData("./dataset/Groceries_dataset.txt");
    // test("fp", kaggleData, 0.01, 0.001);
    // test("ap", kaggleData, 0.01, 0.001);

}