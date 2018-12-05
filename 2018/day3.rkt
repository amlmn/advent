#lang racket

(define (parse s) (cdr (regexp-match #px"#(\\d+) @ (\\d+),(\\d+): (\\d+)x(\\d+)" s)))
(define claims (map (curry map string->number) (map parse (file->lines "input.day3" #:line-mode 'any))))
(define test-claims (map (curry map string->number) (map parse (file->lines "input.day3.test" #:line-mode 'any))))

; Part 1
(define (add-point! points point)
  (define claims (+  1 (hash-ref points point 0)))
  (hash-set! points point claims))
(define (claim->points claim)
  (match-define (list i x y w h) claim)
  (cartesian-product (map (curry + x) (range w)) (map (curry + y) (range h))))
(define (add-claim! points claim) (for-each (curry add-point! points) (claim->points claim)))
(define (part1 claims)
  (define points (make-hash))
  (for-each (curry add-claim! points) claims)
  (length (filter (curry < 1) (hash-values points))))

(displayln (part1 test-claims))
(displayln (part1 claims))

; Part 2
(define (part2 claims)
  (define points (make-hash))
  (for-each (curry add-claim! points) claims)
  (define (no-overlaps? claim) (andmap (lambda (point) (= 1 (hash-ref points point))) (claim->points claim)))
  (ormap (lambda (claim) (if (no-overlaps? claim) (car claim) #f)) claims))

(displayln (part2 test-claims))
(displayln (part2 claims))
