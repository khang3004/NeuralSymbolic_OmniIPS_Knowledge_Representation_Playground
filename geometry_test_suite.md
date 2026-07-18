# GeoIPS — Geometry Verification Test Suite

This test suite contains **15 problems** graded from basic school curriculum to Olympiad/IMO standards. Each problem contains natural language text (to verify GraphRAG) and its equivalent formal predicates/goals (to test the solver directly).

---

## Level 1: Basic (School Curriculum)

### Problem 1: Triangle Interior Angle Sum
*   **Natural Language:** "Cho tam giác ABC. Biết góc A bằng 60 độ, góc B bằng 70 độ. Chứng minh rằng góc C bằng 50 độ."
*   **Formal Inputs:**
    ```json
    [
      "Triangle(A,B,C)",
      "Equal(Angle(B,A,C), 60)",
      "Equal(Angle(A,B,C), 70)"
    ]
    ```
*   **Goal:** `Equal(Angle(B,C,A), 50)`

### Problem 2: SAS Triangle Congruence
*   **Natural Language:** "Cho hai tam giác ABC và DEF. Biết AB = DE, AC = DF và góc A bằng góc D. Chứng minh hai tam giác ABC và DEF bằng nhau."
*   **Formal Inputs:**
    ```json
    [
      "Triangle(A,B,C)",
      "Triangle(D,E,F)",
      "Congruent(Segment(A,B), Segment(D,E))",
      "Congruent(Segment(A,C), Segment(D,F))",
      "Equal(Angle(B,A,C), Angle(E,D,F))"
    ]
    ```
*   **Goal:** `CongruentTriangles(Triangle(A,B,C), Triangle(D,E,F))`

### Problem 3: Parallel Line Corresponding Angles
*   **Natural Language:** "Cho hai đường thẳng AB và CD song song với nhau. Đường thẳng transversial E cắt AC. Chứng minh góc EAB bằng góc ACD."
*   **Formal Inputs:**
    ```json
    [
      "Parallel(AB, CD)",
      "Transversal(E, AB, CD)",
      "Collinear(E, A, C)"
    ]
    ```
*   **Goal:** `Equal(Angle(E,A,B), Angle(A,C,D))`

---

## Level 2: Intermediate (High School / Specialized)

### Problem 4: Cyclic Quadrilateral Opposite Angles
*   **Natural Language:** "Cho tứ giác nội tiếp ABCD nằm trên đường tròn O. Chứng minh rằng tổng hai góc đối diện DAB và BCD bằng 180 độ."
*   **Formal Inputs:**
    ```json
    [
      "Circle(O)",
      "CyclicQuadrilateral(A,B,C,D,Circle(O))"
    ]
    ```
*   **Goal:** `Equal(Add(Angle(D,A,B), Angle(B,C,D)), 180)`

### Problem 5: Midpoint Segment Proportion (Corresponding Angles)
*   **Natural Language:** "Cho tam giác ABC. E là trung điểm của AB, F là trung điểm của AC. Chứng minh đoạn thẳng EF song song với BC."
*   **Formal Inputs:**
    ```json
    [
      "Triangle(A,B,C)",
      "Midpoint(E, Segment(A,B))",
      "Midpoint(F, Segment(A,C))"
    ]
    ```
*   **Goal:** `Parallel(EF, BC)`

### Problem 6: Intersecting Chords Theorem
*   **Natural Language:** "Cho đường tròn O có hai dây cung AB và CD cắt nhau tại điểm P. Chứng minh tích các đoạn thẳng PA * PB bằng PC * PD."
*   **Formal Inputs:**
    ```json
    [
      "Circle(O)",
      "Chord(A, B, Circle(O))",
      "Chord(C, D, Circle(O))",
      "IntersectionPoint(P, Segment(A,B), Segment(C,D))"
    ]
    ```
*   **Goal:** `Equal(Mul(Length(Segment(A,P)), Length(Segment(P,B))), Mul(Length(Segment(C,P)), Length(Segment(P,D))))`

---

## Level 3: Advanced (Specialized Competitions)

### Problem 7: Right Triangle Metric Relation (Geometric Mean)
*   **Natural Language:** "Cho tam giác ABC vuông tại A. H là hình chiếu của A lên cạnh huyền BC. Chứng minh rằng nghịch đảo bình phương đường cao AH bằng tổng nghịch đảo bình phương hai cạnh góc vuông AB và AC."
*   **Formal Inputs:**
    ```json
    [
      "Triangle(A,B,C)",
      "Equal(Angle(B,A,C), 90)",
      "Foot(H, A, Segment(B,C))",
      "Perpendicular(AH, BC)"
    ]
    ```
*   **Goal:** `Equal(Div(1, Pow(Length(Segment(A,H)), 2)), Add(Div(1, Pow(Length(Segment(A,B)), 2)), Div(1, Pow(Length(Segment(A,C)), 2))))`

### Problem 8: Ptolemy's Theorem
*   **Natural Language:** "Cho tứ giác nội tiếp ABCD. Chứng minh rằng tích hai đường chéo AC * BD bằng tổng của tích các cặp cạnh đối diện AB * CD + BC * AD."
*   **Formal Inputs:**
    ```json
    [
      "Circle(O)",
      "CyclicQuadrilateral(A,B,C,D,Circle(O))"
    ]
    ```
*   **Goal:** `Equal(Mul(Length(Segment(A,C)), Length(Segment(B,D))), Add(Mul(Length(Segment(A,B)), Length(Segment(C,D))), Mul(Length(Segment(B,C)), Length(Segment(A,D)))))`

### Problem 9: Tangent-Secant Theorem (Power of a Point)
*   **Natural Language:** "Cho điểm P nằm ngoài đường tròn O. Vẽ cát tuyến PAB và tiếp tuyến PT tới đường tròn. Chứng minh PT bình phương bằng PA nhân PB."
*   **Formal Inputs:**
    ```json
    [
      "Circle(O)",
      "PointOutsideCircle(P, Circle(O))",
      "TangentSegment(P, T, Circle(O))",
      "SecantSegment(P, A, B, Circle(O))"
      
    ]
    ```
*   **Goal:** `Equal(Pow(Length(Segment(P,T)), 2), Mul(Length(Segment(P,A)), Length(Segment(P,B))))`

### Problem 10: Rhombus Diagonals
*   **Natural Language:** "Cho tứ giác ABCD là hình thoi. Chứng minh đường chéo AC vuông góc với đường chéo BD."
*   **Formal Inputs:**
    ```json
    [
      "Rhombus(A,B,C,D)"
    ]
    ```
*   **Goal:** `Perpendicular(Segment(A,C), Segment(B,D))`

---

## Level 4: Olympiad & IMO Standards

### Problem 11: Ceva's Theorem
*   **Natural Language:** "Cho tam giác ABC. Các đường thẳng AD, BE, CF đồng quy tại một điểm trong tam giác (với D trên BC, E trên AC, F trên AB). Chứng minh tích tỉ số các đoạn thẳng (BD/DC) * (CE/EA) * (AF/FB) = 1."
*   **Formal Inputs:**
    ```json
    [
      "Triangle(A,B,C)",
      "PointOnSegment(D, Segment(B,C))",
      "PointOnSegment(E, Segment(A,C))",
      "PointOnSegment(F, Segment(A,B))",
      "Concurrent(Segment(A,D), Segment(B,E), Segment(C,F))"
    ]
    ```
*   **Goal:** `Equal(Mul(Div(Length(Segment(B,D)), Length(Segment(D,C))), Mul(Div(Length(Segment(C,E)), Length(Segment(E,A))), Div(Length(Segment(A,F)), Length(Segment(F,B))))), 1)`

### Problem 12: Menelaus's Theorem
*   **Natural Language:** "Cho tam giác ABC. Một đường thẳng cắt các đường thẳng chứa các cạnh BC, CA, AB lần lượt tại D, E, F sao cho D, E, F thẳng hàng. Chứng minh rằng (BD/DC) * (CE/EA) * (AF/FB) = 1."
*   **Formal Inputs:**
    ```json
    [
      "Triangle(A,B,C)",
      "PointOnLine(D, Segment(B,C))",
      "PointOnLine(E, Segment(A,C))",
      "PointOnLine(F, Segment(A,B))",
      "Collinear(D, E, F)"
    ]
    ```
*   **Goal:** `Equal(Mul(Div(Length(Segment(B,D)), Length(Segment(D,C))), Mul(Div(Length(Segment(C,E)), Length(Segment(E,A))), Div(Length(Segment(A,F)), Length(Segment(F,B))))), 1)`

### Problem 13: Simson Line Theorem
*   **Natural Language:** "Cho tam giác ABC có đường tròn ngoại tiếp O. P là một điểm nằm trên đường tròn O. Gọi X, Y, Z lần lượt là hình chiếu vuông góc của P lên ba cạnh AB, BC, CA. Chứng minh X, Y, Z thẳng hàng."
*   **Formal Inputs:**
    ```json
    [
      "Triangle(A,B,C)",
      "Circle(O)",
      "Circumcircle(Circle(O), Triangle(A,B,C))",
      "PointOnCircle(P, Circle(O))",
      "Foot(X, P, Segment(A,B))",
      "Foot(Y, P, Segment(B,C))",
      "Foot(Z, P, Segment(A,C))"
    ]
    ```
*   **Goal:** `Collinear(X, Y, Z)`

### Problem 14: Midpoint of Median (Varignon's Variant)
*   **Natural Language:** "Cho tứ giác ABCD. Gọi M, N, P, Q lần lượt là trung điểm của AB, BC, CD, DA. Chứng minh MP và NQ cắt nhau tại trung điểm của mỗi đường."
*   **Formal Inputs:**
    ```json
    [
      "Quadrilateral(A,B,C,D)",
      "Midpoint(M, Segment(A,B))",
      "Midpoint(N, Segment(B,C))",
      "Midpoint(P, Segment(C,D))",
      "Midpoint(Q, Segment(D,A))",
      "IntersectionPoint(O, Segment(M,P), Segment(N,Q))"
    ]
    ```
*   **Goal:** `And(Midpoint(O, Segment(M,P)), Midpoint(O, Segment(N,Q)))`

### Problem 15: Nagel Point Precursor Lemma
*   **Natural Language:** "Cho tam giác ABC. Gọi Ta, Tb, Tc lần lượt là tiếp điểm của các đường tròn bàng tiếp góc A, B, C với các cạnh BC, CA, AB. Chứng minh các đoạn thẳng ATa, BTb, CTc đồng quy."
*   **Formal Inputs:**
    ```json
    [
      "Triangle(A,B,C)",
      "ExcirclePoint(Ta, A, Segment(B,C))",
      "ExcirclePoint(Tb, B, Segment(A,C))",
      "ExcirclePoint(Tc, C, Segment(A,B))"
    ]
    ```
*   **Goal:** `Concurrent(Segment(A,Ta), Segment(B,Tb), Segment(C,Tc))`
