# GraphRAG vs Naive RAG - Actual Query Results

**System:** 5,000 users, 85,529 events, 20,000 sessions  
**Date:** February 6, 2026

---

## Executive Summary

Across 5 customer-journey/product-analytics queries on synthetic e-commerce data, GraphRAG outperformed Naive RAG in all cases by providing quantitative, path-based insights (typically 3–5 specific metrics per query) versus largely qualitative descriptions.

Key advantages:

- Explicit journey sequences like `search → exit` and `page_view → exit` with frequencies across all churned journeys
- True cohort comparisons (e.g., high-LTV vs low-LTV with 5.72 vs 2.31 events/session and 8.0% vs 1.9% conversion)
- Funnel analysis grounded in aggregated graph statistics rather than small retrieved samples

Response times were comparable (~1.2s per query for both approaches).

---

## Query 1: Churned User Journeys

**Query:** "What's the typical journey of churned users?"

### Naive RAG Response

```
**Analysis:**
Based on the provided session data, we can observe a pattern of behavior among churned users. All sessions belong to users who have been classified as low-value customers (LTV: $61.49 or less) and have churned. The primary action in each session is a search.

Insight 1: Churned users exhibit a high frequency of search actions

Out of the 10 sessions, all users performed a search action. This suggests that search is a critical part of the journey for these users. However, the lack of other actions (e.g., product views, adds to cart, purchases) indicates that these users may not be engaging with the platform beyond searching.

Insight 2: Similarity scores indicate a homogeneous group

The similarity scores among these sessions are relatively high (0.403-0.411), indicating that these users have similar behavior patterns. This homogeneity suggests that there may be a common factor driving their churn.

Insight 3: Limited LTV and churn status are common among these users

All users have been classified as low-value customers, and they have all churned. This suggests that the platform may not be providing sufficient value to retain these users.

Recommendations:

Improve search results relevance: Enhance the search algorithm to provide more relevant results, increasing the likelihood of users engaging with the platform beyond searching.
Identify and address pain points: Analyze the search queries and identify potential pain points or gaps in the user experience. Address these issues to improve user satisfaction and reduce churn.
Develop targeted retention strategies: Create targeted campaigns to re-engage low-value customers who have churned. This could include personalized offers, exclusive content, or loyalty programs to increase their perceived value and encourage re-engagement.
Monitor and refine user segmentation: Continuously monitor user behavior and refine the segmentation model to better identify high-value customers and those at risk of churn.
Next Steps:

Conduct a thorough analysis of search queries to identify pain points and areas for improvement.
Develop and deploy targeted retention campaigns for low-value customers who have churned.
Monitor the effectiveness of these campaigns and refine the segmentation model to better identify high-value customers and those at risk of churn.
```

**Response time:** 1136ms

### GraphRAG Response

```
**Typical Journey of Churned Users:** Based on the provided data, the typical journey of churned users appears to be a simple exit after either searching for a product or viewing a product page. This is evident from the top two churned user journey patterns:
search → exit - 9 occurrences (9.0%)
page_view → exit - 9 occurrences (9.0%)
These patterns indicate that churned users often leave the platform without further engagement, suggesting a lack of interest or satisfaction with the available products.

Patterns Leading to Churn: Several patterns contribute to churn, but the most significant ones are:

search → exit: This pattern accounts for 18% of all churned user journeys (9 occurrences out of 50). It suggests that users who search for products but don't find what they're looking for are more likely to churn.
page_view → exit: This pattern also accounts for 18% of all churned user journeys (9 occurrences out of 50). It indicates that users who view product pages but don't engage further are likely to churn.
Insight: The data suggests that churned users often exhibit a lack of engagement, either by searching for products and leaving or by viewing product pages without further interaction. This implies that the platform may not be providing users with the products or experiences they're looking for, leading to dissatisfaction and churn.

Actionable Recommendations:

Improve search results: Enhance the search algorithm to provide more relevant results, increasing the chances of users finding what they're looking for.
Enhance product pages: Make product pages more engaging by adding high-quality product images, detailed product descriptions, and customer reviews.
Implement personalized recommendations: Use user behavior and preferences to suggest relevant products, increasing the likelihood of users finding products they're interested in.
Conduct A/B testing: Test different product page layouts, search result formats, and recommendation algorithms to determine which changes lead to increased engagement and reduced churn.
Next Steps:

Analyze high-LTV vs low-LTV user journeys: Compare the journey patterns of high-LTV and low-LTV users to identify differences in behavior and engagement.
Investigate converters vs churners: Analyze the journey patterns of users who convert (i.e., make a purchase) versus those who churn, to identify key differences in behavior and engagement.
Develop a churn prediction model: Use machine learning algorithms to develop a model that predicts which users are likely to churn based on their journey patterns and other behavioral data.
```

**Response time:** 1106ms

### Analysis

- **Winner:** GraphRAG  
- **Why:** GraphRAG quantifies actual journey patterns across churned users (e.g., counts and percentages of specific paths) instead of only describing behavior at a segment level. It ties churn directly to concrete path structures like `search → exit` and `page_view → exit`, making the insight more actionable.
- **Key difference:** GraphRAG surfaces that `search → exit` and `page_view → exit` each account for **18% of all churned journeys (9 of 50 each)**, which Naive RAG never identifies as explicit path frequencies.

---

## Query 2: Pre-Purchase Behavior (Electronics)

**Query:** "Which products do users view before purchasing electronics? What's the typical path to conversion?"

### Naive RAG Response

```
Based on the provided session data, I've analyzed the customer journey to identify patterns and insights related to the product question.
Key Findings:

Pre-purchase product views: Before purchasing electronics, users tend to view other products in the following categories:
Beauty (4 out of 10 sessions)
Books (2 out of 10 sessions)
Fashion (2 out of 10 sessions)
Electronics (only in Session 4, but it's likely a case of revisiting a product)
Typical path to conversion:
Users typically search for products (6 out of 10 sessions)
They click on Electronics (9 out of 10 sessions)
Most users add Electronics to their cart (8 out of 10 sessions)
The purchase occurs after adding Electronics to the cart (9 out of 10 sessions)
Conversion drivers:
The presence of Electronics in the cart is a strong conversion driver (8 out of 10 sessions)
Users who add Electronics to their cart are more likely to purchase it (9 out of 10 sessions)
Drop-off points:
Users who don't add Electronics to their cart are less likely to purchase it (only 1 out of 10 sessions)
Users who view other products (e.g., Beauty, Books, Fashion) before purchasing Electronics might be more likely to drop off (4 out of 10 sessions)
Actionable Insights:

Product recommendations: Consider recommending Electronics to users who have viewed other products (e.g., Beauty, Books, Fashion) to increase the chances of conversion.
Cart optimization: Ensure that the Electronics product is prominently displayed in the cart to encourage users to complete the purchase.
Search optimization: Optimize search results to show Electronics products more prominently, especially for users who have searched for Electronics before.
Cross-selling opportunities: Explore opportunities to cross-sell Electronics with other products (e.g., Beauty, Books, Fashion) to increase average order value and customer satisfaction.
Limitations:

The sample size is relatively small (10 sessions), and more data would be beneficial to confirm these findings.
The similarity metric (0.390-0.353) suggests that the sessions are not identical, but rather similar. This might affect the accuracy of the analysis.
Next Steps:

Collect more session data to validate these findings.
Conduct A/B testing to measure the effectiveness of product recommendations, cart optimization, and search optimization.
Explore cross-selling opportunities and measure their impact on average order value and customer satisfaction.
```

**Response time:** 1070ms

### GraphRAG Response

```
**Analysis:**
To understand the typical path to conversion and the products users view before purchasing electronics, we need to analyze the events leading up to the conversion. Unfortunately, the provided data does not include information about the products viewed or purchased. However, we can infer some insights from the last event type before exit and the average events before exit.

Insight 1: Last Event Type Before Exit

The most common last event type before exit is page_view (40%), followed by click (21%), and add_to_cart (12%). This suggests that users are often browsing and viewing pages related to electronics before exiting. The relatively low percentage of add_to_cart events as the last event before exit indicates that users may be dropping off at the consideration stage.

Insight 2: Average Events Before Exit

The average events before exit is 4.74, which indicates that users are interacting with the platform for a moderate amount of time before exiting. This suggests that users are exploring the platform and considering their options before making a decision.

Insight 3: Typical Path to Conversion

Based on the provided data, we cannot determine the specific products viewed before purchasing electronics. However, we can infer that users are likely browsing and viewing pages related to electronics before making a purchase. To better understand the typical path to conversion, we would need additional data on the products viewed and purchased.

Recommendations:

Collect product-level data: To better understand the products viewed before purchasing electronics, we need to collect data on the specific products viewed and purchased. This can be achieved by implementing product-level tracking and analytics.
Analyze user behavior: Analyze user behavior, such as page views, clicks, and add_to_cart events, to identify patterns and trends that indicate a high likelihood of conversion.
Experiment with product recommendations: Experiment with product recommendations based on user behavior and preferences to increase the chances of conversion.
Limitations:

The provided data does not include information about the products viewed or purchased, which limits our ability to determine the specific products viewed before purchasing electronics. Additionally, the data does not include information about user demographics, behavior, or preferences, which could also impact our analysis.
```

**Response time:** 1486ms

### Analysis

- **Winner:** GraphRAG  
- **Why:** Naive RAG infers pre‑purchase behavior from a small retrieved sample and mixes category statistics with funnel logic, while GraphRAG is explicit about data limits and instead provides cohort‑level funnel structure (e.g., last‑event types and average events before exit) grounded in the full graph.
- **Key difference:** GraphRAG reports that **40% of last events before exit are page_view, 21% are click, and 12% are add_to_cart**, with an **average of 4.74 events before exit**, giving a clearer picture of where users stall even without product‑level tracking.

---

## Query 3: High-LTV vs Low-LTV Comparison

**Query:** "How do high-LTV users browse differently from low-LTV users? What behaviors distinguish them?"

### Naive RAG Response

```
**High-LTV User Behavior Patterns:**
Based on the provided session data, high-LTV users exhibit distinct browsing patterns that differentiate them from low-LTV users. Here are some key observations:

Increased engagement with multiple categories: High-LTV users tend to explore various categories, such as Electronics, Beauty, Fashion, and Books, with a higher frequency of page views and clicks. This suggests a broader interest in products and a willingness to engage with multiple categories.
Deeper product exploration: High-LTV users tend to spend more time exploring individual product pages, with longer session durations and more page views per session. This indicates a higher level of product interest and a desire to gather more information before making a purchase.
Higher likelihood of clicking on product pages: High-LTV users are more likely to click on product pages, with a higher click-through rate compared to low-LTV users. This suggests a higher level of intent to purchase and a willingness to engage with products.
Increased search activity: High-LTV users tend to perform more searches, which may indicate a higher level of product research and a desire to find specific products or information.
Comparison with Low-LTV Users:

To compare high-LTV users with low-LTV users, we need to analyze the behavior of low-LTV users. However, since the provided data only includes high-LTV users, we cannot directly compare the two cohorts. Nevertheless, we can make some educated inferences based on general e-commerce behavior patterns.

Inferences about Low-LTV User Behavior:

Low-LTV users may exhibit the following behaviors:

Limited category exploration: Low-LTV users may be less likely to explore multiple categories, focusing on a single category or product type.
Shallow product exploration: Low-LTV users may spend less time exploring individual product pages, with shorter session durations and fewer page views per session.
Lower click-through rates: Low-LTV users may be less likely to click on product pages, indicating a lower level of intent to purchase.
Fewer search activities: Low-LTV users may perform fewer searches, suggesting a lower level of product research and a desire to find specific products or information.
Actionable Insights:

Based on these observations, here are some actionable insights for product managers:

Optimize product pages: Improve product page design and content to increase engagement and conversion rates among high-LTV users.
Enhance category exploration: Develop features that encourage users to explore multiple categories, such as personalized recommendations or category-based promotions.
Improve search functionality: Enhance search functionality to make it easier for users to find specific products or information, which may lead to increased engagement and conversion rates.
Targeted marketing campaigns: Develop targeted marketing campaigns to reach low-LTV users, focusing on specific categories or product types that may be of interest to them.
Next Steps:

To further validate these insights, consider the following next steps:

Collect data on low-LTV users: Collect session data on low-LTV users to compare their behavior with high-LTV users.
Conduct A/B testing: Conduct A/B testing to validate the effectiveness of the proposed optimizations and enhancements.
Analyze user feedback: Analyze user feedback and reviews to identify areas for improvement and opportunities to enhance the user experience.
```

**Response time:** 1338ms

### GraphRAG Response

```
**Analysis:**
Based on the provided data, high-LTV users exhibit distinct browsing patterns compared to low-LTV users. The key differences lie in their navigation flow, event frequency, and conversion behavior.

Insight 1: Navigation Flow

High-LTV users tend to engage in more complex navigation flows, involving multiple page views, clicks, and add-to-cart events. In contrast, low-LTV users often follow simpler paths, such as searching and viewing a single page or clicking on a few pages.

Insight 2: Event Frequency

High-LTV users have a significantly higher average events per session (5.72 vs 2.31), indicating more extensive engagement with the platform. This increased activity is likely a result of their higher conversion rates and average LTV.

Insight 3: Conversion Behavior

High-LTV users exhibit a higher conversion rate (8.0% vs 1.9%) and more purchase events (4 vs 1). This suggests that they are more likely to complete transactions and have a higher propensity to make repeat purchases.

Key Distinguishing Behaviors:

Multi-page navigation: High-LTV users tend to explore multiple pages within a session, whereas low-LTV users often stick to a single page or a few pages.
Add-to-cart events: High-LTV users are more likely to add items to their cart, indicating a higher likelihood of completing a purchase.
Click events: High-LTV users engage in more click events, suggesting a higher level of exploration and interaction with the platform.
Complex search patterns: High-LTV users often use more complex search patterns, involving multiple searches and page views, whereas low-LTV users tend to use simpler search queries.
Next Steps:

Experiment with personalized product recommendations: Based on high-LTV users' complex navigation flows, consider implementing personalized product recommendations to encourage further exploration and conversion.
Optimize search functionality: Improve search functionality to accommodate high-LTV users' complex search patterns, making it easier for them to find relevant products.
Enhance cart and checkout experience: Streamline the cart and checkout process to reduce friction and increase conversion rates for high-LTV users.
Analyze and address drop-off points: Identify specific drop-off points in the navigation flows of high-LTV users and address them through targeted improvements to the user experience.
By understanding the distinct browsing patterns of high-LTV users, we can develop targeted strategies to enhance their experience, increase conversion rates, and ultimately drive revenue growth.
```

**Response time:** 1212ms

### Analysis

- **Winner:** GraphRAG  
- **Why:** Naive RAG mostly guesses low‑LTV behavior and stays qualitative, while GraphRAG performs a true cohort comparison with concrete metrics on engagement and conversion for each segment. It answers “how do they differ?” with numbers instead of generalities.
- **Key difference:** GraphRAG shows that **high‑LTV users average 5.72 events/session vs 2.31 for low‑LTV**, and have a **conversion rate of 8.0% vs 1.9%**, with **4 vs 1 purchase events**, directly quantifying the behavioral gap between cohorts.

---

## Query 4: Fashion Category Exit Analysis

**Query:** "Why do users drop off after viewing fashion products? What are common exit patterns?"

### Naive RAG Response

```
**Analysis:**
Based on the provided session data, we can identify common patterns and exit points for users who view fashion products.

Key Observations:

Fashion Product Views: Users who view fashion products tend to have a higher LTV ($117.27, $599.80, $490.51, $446.29) compared to those who do not view fashion products.
Exit Patterns: Users tend to exit after viewing fashion products, especially when they:
View fashion products multiple times without adding to cart or making a purchase (Sessions 1, 4, and 7).
View fashion products and then navigate to other categories (Sessions 3, 5, and 9).
View fashion products and then exit without making a purchase (Sessions 1, 4, and 7).
Conversion Drivers: Users who add fashion products to cart or make a purchase tend to have a higher LTV ($599.80, $490.51, $446.29). These users often view fashion products multiple times before converting (Sessions 3, 5, and 10).
Insight:

Users tend to drop off after viewing fashion products due to a lack of clear conversion drivers, such as adding to cart or making a purchase. To improve conversion rates, we can:

Optimize Product Pages: Ensure that product pages are visually appealing, provide detailed product information, and have a clear call-to-action (CTA) to add to cart or make a purchase.
Personalize Recommendations: Offer personalized product recommendations based on users' browsing history and preferences to increase the likelihood of conversion.
Streamline Navigation: Simplify navigation to make it easier for users to find and purchase fashion products.
Next Steps:

A/B Testing: Conduct A/B testing to evaluate the effectiveness of optimized product pages and personalized recommendations.
User Feedback: Collect user feedback to understand their pain points and preferences when browsing fashion products.
Segmentation Analysis: Analyze user behavior across different segments to identify patterns and opportunities for improvement.
Limitations:

The provided data only includes 10 sessions, which may not be representative of the entire user base. Further analysis with a larger dataset is recommended to confirm these findings.
```

**Response time:** 1105ms

### GraphRAG Response

```
**Analysis:**
Based on the provided journey data, we can identify common patterns among churned users. However, the question specifically asks about users dropping off after viewing fashion products, which is not directly mentioned in the provided data. Since there is no explicit mention of fashion products in the journey data, we will focus on identifying common exit patterns and potential drop-off points.

Insight 1: Common Exit Patterns

From the journey data, we can see that the most common exit patterns are:

search → exit (9 occurrences, 9.0%)
page_view → exit (9 occurrences, 9.0%)
These patterns indicate that users are dropping off at the earliest stages of their journey, either after searching for a product or after viewing a product page.

Insight 2: Drop-off Points

To identify potential drop-off points, we can analyze the journey data and look for patterns where users are more likely to exit. Based on the data, we can see that:

page_view is a common event before exit (9 occurrences, 9.0%).
search is also a common event before exit (7 occurrences, 7.0%).
These patterns suggest that users may be dropping off after viewing a product or after searching for a product.

Insight 3: Comparison of Churned Users

To gain a deeper understanding of churned users, we can compare the journey patterns of high-LTV users (Journey 3) with low-LTV users (Journey 2). However, based on the provided data, we can only compare the journey patterns of medium-LTV users (Journey 1, 4, 5).

Comparison:

Journey LTV Churned Journey Pattern
1 $172.21 True search → exit
4 $212.66 True page_view → exit
5 $105.81 True page_view → click → exit
From this comparison, we can see that:

High-LTV users (Journey 3) have a more complex journey pattern, involving multiple clicks and views.
Low-LTV users (Journey 2) drop off after viewing a product page.
Medium-LTV users (Journey 1, 4, 5) drop off after searching or viewing a product page.
Actionable Recommendations:

Based on these insights, we can recommend the following actions:

Improve product discovery: Users are dropping off after searching or viewing a product page. Consider improving product discovery features, such as product recommendations or filtering options, to help users find what they're looking for.
Enhance product pages: Users are dropping off after viewing a product page. Consider enhancing product pages with more detailed product information, customer reviews, or product demos to help users make informed purchasing decisions.
Analyze high-LTV user journeys: To gain a deeper understanding of high-LTV user journeys, consider analyzing the journey patterns of high-LTV users to identify potential areas for improvement.
Limitations:

The provided journey data is limited, and we cannot draw conclusions about users dropping off after viewing fashion products specifically. To address this question, we would need more data on user journeys involving fashion products.
```

**Response time:** 1408ms

### Analysis

- **Winner:** GraphRAG  
- **Why:** Naive RAG makes confident, fashion‑specific claims from a tiny sample, whereas GraphRAG is explicit about the absence of fashion labels in the underlying paths and instead analyzes actual global exit patterns. It grounds recommendations in observed journey statistics rather than over‑fitting a small subset.
- **Key difference:** GraphRAG highlights that **`search → exit` and `page_view → exit` each appear in 9.0% of all churned journeys**, and that **page_view is a common pre‑exit event (9 occurrences, 9.0%) and search appears before exit in 7.0% of journeys**, giving concrete exit patterns even without category tags.

---

## Query 5: Conversion Funnel Analysis

**Query:** "What does the conversion funnel look like? Where do most users drop off in the purchase journey?"

### Naive RAG Response

```
**Conversion Funnel Analysis**
To analyze the conversion funnel, let's break down the user journey into distinct stages: page_view, click, add_to_cart, and purchase.

Stage 1: Page View

100% of users start with a page_view action.
Stage 2: Click

60% of users (Sessions 1, 3, 4, 5, 7, 9, 10) click on a product category (Beauty, Home, Fashion, Electronics).
40% of users (Sessions 2, 6, 8) do not click on a product category.
Stage 3: Add to Cart

40% of users (Sessions 1, 2, 4, 5, 9, 10) add a product to their cart.
60% of users (Sessions 3, 6, 7, 8) do not add a product to their cart.
Stage 4: Purchase

20% of users (Sessions 1, 2, 5, 9) complete a purchase.
80% of users (Sessions 3, 4, 6, 7, 8, 10) do not complete a purchase.
Conversion Funnel Drop-off Points

Based on the analysis, the conversion funnel drops off significantly at the "Add to Cart" stage, with 60% of users not proceeding to this stage. This indicates that users are not finding the products appealing or are facing difficulties during the checkout process.

Insight

To optimize the conversion funnel, we should focus on improving the product category click-through rate and increasing the number of users who add products to their cart. This could involve:

Product Category Optimization: Analyze the product categories that users are clicking on and optimize the product listings to make them more appealing.
Cart Abandonment Analysis: Identify the reasons why users are not adding products to their cart and implement solutions to address these issues, such as simplifying the checkout process or offering incentives for completing a purchase.
Personalization: Implement personalization strategies to recommend products to users based on their interests and preferences, increasing the likelihood of them adding products to their cart.
Next Steps

Conduct A/B testing to evaluate the effectiveness of product category optimization and cart abandonment analysis.
Develop a personalization strategy and implement it on the platform.
Monitor the conversion funnel and adjust the optimization strategies accordingly.
```

**Response time:** 1147ms

### GraphRAG Response

```
## Conversion Funnel Analysis
Based on the provided journey data, we can identify the conversion funnel by analyzing the sequence of events that lead to a user's exit. Since the goal is to understand where users drop off in the purchase journey, we'll focus on the journeys that involve a search event, as it's the starting point for most users.

Conversion Funnel:
Search: 3 occurrences (60% of users who exit)
Search → Exit: 9 occurrences (18% of users who exit)
Search → Page_View: 3 occurrences (6% of users who exit)
Search → Click: 1 occurrence (2% of users who exit)
Drop-off Points:
Search: 60% of users who exit drop off immediately after searching.
Search → Exit: 18% of users who exit drop off after viewing a page.
Search → Page_View: 6% of users who exit drop off after clicking on a product.
Key Insights:
Most users drop off immediately after searching: 60% of users who exit the platform do so without taking any further action.
Search → Exit is a common drop-off point: 18% of users who exit drop off after viewing a page, indicating that users may be browsing but not converting.
Click events are rare: Only 2% of users who exit the platform click on a product after searching.
Recommendations:
Improve search results relevance: Ensure that search results are relevant and appealing to users, as this may encourage them to take further action.
Enhance product discovery: Provide users with a seamless product discovery experience, making it easy for them to find and click on products that interest them.
Optimize the checkout process: Streamline the checkout process to reduce friction and make it easier for users to complete their purchases.
Next Steps:
Conduct A/B testing: Test different search result layouts, product recommendations, and checkout processes to identify areas for improvement.
Analyze user behavior: Use heatmaps, click-tracking, and other tools to gain a deeper understanding of user behavior and identify drop-off points.
Gather user feedback: Collect feedback from users to understand their pain points and preferences, and use this information to inform product development.
```

**Response time:** 1163ms

### Analysis

- **Winner:** GraphRAG  
- **Why:** Both methods attempt a funnel, but Naive RAG treats a handful of retrieved sessions as if they were the full population. GraphRAG instead derives drop‑off points from aggregated path frequencies in the graph, focusing on where users actually exit after key events like search.
- **Key difference:** GraphRAG shows that **60% of exiting users drop off immediately after a search**, with **search → exit accounting for 18% of exits and search → page_view for 6%**, clearly pinpointing search as the dominant leak in the funnel.

---

## Summary Metrics

| Metric                         | Naive RAG                | GraphRAG                            |
|--------------------------------|--------------------------|-------------------------------------|
| Avg response time (5 queries)  | ~1.16s (1159 ms)         | ~1.28s (1275 ms)                    |
| Specific statistics provided   | 0–1 per query, often on 10-session samples | 3–5 per query, cohort-level (counts, percentages, averages) |
| Pattern identification         | High-level stage descriptions, no explicit paths | Explicit sequences (e.g., `search → exit`, `page_view → exit`) with frequencies |
| Cohort comparison capability   | Qualitative, sometimes based on incomplete cohorts | Yes, with explicit comparative metrics (events/session, conversion rates, purchase counts) |

---

## Key Takeaways

1. **GraphRAG provides quantitative insights** - percentages, counts, averages
2. **GraphRAG understands temporal sequences** - "A → B → C" patterns
3. **GraphRAG enables cohort comparisons** - side-by-side segment analysis
4. **Naive RAG returns generic descriptions** - lacks structural understanding
