/****** Script for SelectTopNRows command from SSMS  ******/
SELECT DISTINCT(Label), Label_ID, NIP, Amount FROM [MMandziej].[dbo].[PWG_finance_data]
WHERE UPPER(Label) LIKE '%ZYSK (STRATA) NET%' AND UPPER(Label) LIKE '%operacyjnej%'

SELECT * FROM [MMandziej].[dbo].[PWG_finance_data]
WHERE Label LIKE '%I. Kapital (fundusz) podstawowy%' 


SELECT * FROM [MMandziej].[dbo].[PWG_finance_data]
WHERE Label LIKE '%Al%'

UPDATE dbo.PWG_finance_data SET [Label] = dbo.Format_RemoveAccents([Label])