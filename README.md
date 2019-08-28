# impute-gender

## notes

The gender most commonly associated with a given name can change considerably over time. In the United States, Hillary, Leslie, and Jordan are common names that have shifted over the last century from being predominantly "boys' names" to almost exclusively "girls' names."


#### List:

                         % Female
 Name           1920    1950    1980    2010   Slope   Overall        n
--------------------------------------------------------------------------
Anne
Donna
Elizabeth       99.7    99.7    99.7    99.7     0      99.7    1,587,027
Hillary
Jordan
Leslie
Matthew         0.4     0.4     0.4     0.4      0       0.4    1,532,144
Thomas

Date of birth is therefore a useful -- but by no means perfect -- feature to consider when imputing gender from a name. (fn: Family national or cultural background can also be beneficial. Among Russian families, newborn Alexis is likely male, but we'd guess female in recent United States births. --also Andreas? -- Robin in England vs U.S.--)

#### Graph:


              Low variation
                  /  \
                 /    \   Matthew
                /      \
               /        \
              /          \   Jordan
             /            \
            /              \
           / High variation \
          /                  \   Hillary
         /                    \
Time-Independent       Time-Dependent


For any specific name, a logistic regression that captures the functional form of the relationship between year and gender ratio could be suitable. However, as above, this is a name-by-name process. Creating a robust specification for every name in the SSA database would require some 93xxx regressions, each with a potentially different relationship. Some might be approximated by a quadratic term, others would be more accurately captured by a spline.

#### small multiples graph with various weird versions of time-dependent names


## gender data

Demonstration uses data from the wonderful https://github.com/OpenGenderTracking/globalnamedata

Need to compare https://bocoup.com/blog/global-name-data to
https://github.com/ropensci/genderdata


## other projects

https://experts.illinois.edu/en/publications/a-search-engine-approach-to-estimating-temporal-changes-in-gender

https://databank.illinois.edu/datasets/IDB-9087546

http://abel.ischool.illinois.edu/cgi-bin/genni/nameprediction.cgi?name=Leslie

https://cran.r-project.org/web/packages/gender/vignettes/predicting-gender.html

https://github.com/ropensci/gender
 - uses name and year range; only provides point estimates
 - Cameron Blevins and Lincoln Mullen, "Jane, John ... Leslie? A
Historical Method for Algorithmic Gender Prediction," _Digital
Humanities Quarterly_ 9, no. 3 (2015): <http://www.digitalhumanities.org/dhq/vol/9/3/000223/000223.html>.

https://github.com/jeremybmerrill/beauvoir
 - only uses name; only provides point estimates

https://github.com/cblevins/Gender-ID-By-Time
 - ancestor of R's `gender`
